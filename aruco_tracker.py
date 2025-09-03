from enum import Enum

import robotic as ry
import numpy as np
import cv2.aruco


class Smoothing(Enum):
    LAST = 0
    AVERAGE = 1
    FILTER = 2


aruco_6x6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


class ArucoTracker():
    """
    Track aruco markers by using the output of a rgb-d camera.

    Positions are updated every time a marker is discovered.
    Track markers by using the track_markers method.

    Marker positions can be accessed by their id (tracker[id]). This returns None
    if the marker was not observed so far.
    Alternativly, marker_positions provides a dictionary of all known marker positions.

    Args:
        C: The configuration object.
        camera_frame: The camera frame to use.
        marker_dict: The ArUco marker dictionary.
        position_smoothing: How the new measurement updates the marker position.
            Options are:
                - LAST: Only the last measurement is used.
                - AVERAGE: The average of all measurements is used.
                - MOMENTUM: A momentum-based/lowpass approach is used.
        momentum: The momentum factor for the smoothing. The new position is
            calculated as momentum * old_position + (1 - momentum) * new_position
    """
    def __init__(self,
            C: ry.Config,
            camera_frame: str | ry.Frame = 'l_cameraWrist',
            marker_dict: cv2.aruco_Dictionary = aruco_6x6,
            position_smoothing: Smoothing = Smoothing.LAST,
            filter_alpha: float = 0.95
        ):
        self.C = C
        if type(camera_frame) is str:
            self.camera_frame = C.getFrame(camera_frame)
        self.aruco_dict = marker_dict
        self.position_smoothing = position_smoothing
        self.filter_alpha = filter_alpha # TODO: better name for momentum?
        self.marker_visits = dict() # used to calculate the average
        self.marker_positions = dict()
        self.marker_names = dict()

    def _camera_to_global_pos(self, position):
        """Transforms a position in camera coordinates to global coordinates."""
        homogenous_coord = np.concatenate([position, np.ones(1)])
        global_pos = self.camera_frame.getTransform() @ homogenous_coord
        return global_pos[:3] / global_pos[3] # is that needed?

    def _calculate_marker_position(self, corners, points):
        """calculates the position of a detected marker."""
        corner_pixels = corners.squeeze().astype("int")
        corner_points = points[corner_pixels[:,1],corner_pixels[:,0]]
        rel_position = np.average(corner_points, axis=0)
        return self._camera_to_global_pos(rel_position)

    def _update_marker(self, id, position):
        """Updates the position of a detected marker."""
        if id not in self.marker_positions:
            self.marker_positions[id] = position
            self.marker_visits[id] = 1
            return
        
        match self.position_smoothing:
            case Smoothing.LAST:
                self.marker_positions[id] = position
            case Smoothing.AVERAGE:
                self.marker_positions[id] = (self.marker_positions[id] * self.marker_visits[id] + position) / (self.marker_visits[id] + 1)
                self.marker_visits[id] += 1
            case Smoothing.FILTER:
                self.marker_positions[id] = self.filter_alpha * self.marker_positions[id] + (1 - self.filter_alpha) * position

    def track_markers(self, 
            rgb: np.ndarray, 
            points: np.ndarray
        ) -> list[int]:
        """Tracks aruco markers in the given RGB image.
        Args:
            rgb: The RGB image from the camera as a [h,w,3] array.
            points: The 3D points corresponding to the RGB image as a [h,w,3] array.

        Returns:
            List of ids of the markers visible in the image.
        """
        corners, ids, _ = cv2.aruco.detectMarkers(rgb, self.aruco_dict)

        if ids is not None:
            for corner, id in zip(corners, ids.flatten()):
                position = self._calculate_marker_position(corner, points)
                self._update_marker(id, position)
            return list(ids.flatten())

        return []

    def tag_markers(self,
                    ids: int | list[int] | None = None,
                    name: str = "marker",
                    shape: ry.ST = ry.ST.marker):
        """places a frame at every marker.

        Args:
            ids: List of Ids to tag, if None, tags all known markers
            name: The name of the frames. Frames are named name_id.
            shape: The shape of the frames to create.
        """
        if ids is None:
            ids = self.marker_positions.keys()
        elif type(ids) is int:
            ids = [ids]
        for marker_id in ids:
            position = self.marker_positions.get(marker_id)
            if position is not None:
                frame = self.C.addFrame(f"{name}_{marker_id}")
                frame.setShape(shape, [.2])
                frame.setPosition(position)
                self.marker_names[marker_id] = f"{name}_{marker_id}"

    def get_tracked_markers(self) -> list[int]:
        """Returns a list of all tracked marker IDs."""
        return list(self.marker_positions.keys())

    def get_marker_names(self, ids: int | list[int]) -> str | None:
        """
        Returns the name of the marker if it was tagged.

        Args:
            id: The ID or list of IDs of the marker to get the name for.

        Returns:
            For a single ID the function returns the name of the marker if it was tagged, None otherwise.
            For a list of IDs the function returns a list of names the names of tagged markers 
            corresponding to the ids (empty if no markers where tagged).
        """
        if isinstance(ids, list):
            return [self.marker_names[id] for id in ids if id in self.marker_names]
        return self.marker_names.get(ids, None)

    def drop_marker(self, id: int) -> bool:
        """
        Removes a marker from tracking.

        Args:
            id: The ID of the marker to remove.

        Returns:
            True if the marker was tracked beforehand, False otherwise.
        """
        if id in self.marker_positions:
            del self.marker_positions[id]
            del self.marker_visits[id]
            return True
        return False
    

    def reset_average(self):
        """
        Resets the average position of all tracked markers.
        This treats the current tracked position as the first measurement
        """
        for id in self.marker_positions:
            self.marker_visits[id] = 1

    def __getitem__(self, item: int) -> np.ndarray | None:
        """Returns the position of the marker with the given id
        if it was tracked and None otherwise."""
        return self.marker_positions.get(item, None)
    

    def __len__(self):
        return len(self.marker_positions)





if __name__ == "__main__":
    # small example showing the tracked markers
    
    smoothing = Smoothing.LAST # change to AVERAGE or MOMENTUM to see the difference
    
    C = ry.Config()
    C.addFile(ry.raiPath('scenarios/pandaSingle_camera.g'))

    bot = ry.BotOp(C, True)

    pclFrame = C.addFrame('pcl', 'l_cameraWrist')

    bot.hold(floating=True, damping=False)

    tracker = ArucoTracker(C, position_smoothing=Smoothing.LAST)

    while bot.getKeyPressed()!=ord('q'):
        rgb, depth, points = bot.getImageDepthPcl('l_cameraWrist')

        pclFrame.setPointCloud(points, rgb)

        tracker.track_markers(rgb, points)

        tracker.tag_markers(name="marker", shape=ry.ST.marker)

        bot.sync(C, .1, 'ArucoTracker demo')