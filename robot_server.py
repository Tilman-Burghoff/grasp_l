import cv2
import zmq
import pickle
import numpy as np
import robotic as ry
import traceback


class RobotServer:

    def __init__(self, address: str="tcp://*:1234", on_real: bool=False, verbose: int=0):
        
        self.C = ry.Config()
        self.C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))
        self.C.view(False)
        self.bot = ry.BotOp(self.C, on_real)
        self.bot.home(self.C)
        self.verbose = verbose
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.address = address
        self.socket.bind(address)
        self.bot.gripperClose(ry._left)



    def send_error_message(self, text):
        message = {}
        message["success"] = False
        message["text"] = text
        to_send = pickle.dumps(message)
        self.socket.send(to_send)

    def execute_command(self, message: dict) -> dict:

        feedback = {}
        command = message["command"]
        if command == "move":
        
            self.bot.move(message["path"], message["times"])
            while self.bot.getTimeToEnd() > 0:
                self.bot.sync(self.C)

        elif command == "moveTo":
        
            self.bot.moveTo(message["q"], message["time_cost"])
            while self.bot.getTimeToEnd() > 0:
                self.bot.sync(self.C)

        elif command == "moveAutoTimed":
        
            self.bot.moveAutoTimed(message["path"], message["time_cost"])
            while self.bot.getTimeToEnd() > 0:
                self.bot.sync(self.C)

        elif command == "home":
            self.bot.home(self.C)

        elif "gripper" in command:
            which = ry._right if message["gripper_id"] == "right" else ry._left
            
            if "close" in command:
                self.bot.gripperClose(which)
            elif "open" in command:
                self.bot.gripperMove(which)
            else:
                raise Exception(f"Gripper command {message['command']} not implemented.")
            
            while not self.bot.gripperDone(which):
                self.bot.sync(self.C)

        elif command == "getImageDepthPcl":
            rgb, depth, point_cloud = self.bot.getImageDepthPcl(message["sensor_name"])
            feedback["rgb"] = rgb
            feedback["depth"] = depth
            feedback["point_cloud"] = point_cloud
        
        elif command == "getCameraFxycxy":
            Fxycxy = self.bot.getCameraFxycxy(message["sensor_name"])
            feedback["Fxycxy"] = Fxycxy
        
        else:
            raise Exception(f"Command {message['command']} not implemented.")
        
        return feedback
        
    def run(self):

        if self.verbose:
            print("Started server at ", self.address)

        running = True
        while running:
            client_input = self.socket.recv()
            try:
                client_input = pickle.loads(client_input)
            except Exception as e:
                print()
                traceback.print_exc()
                print()
                self.send_error_message(f"Error while loading message: {e}")
                
            if self.verbose:
                print(f"Received request: {client_input}")

            try:
                if client_input["command"] == "close":
                    running = False
                else:
                    feedback = self.execute_command(client_input)

            except Exception as e:
                print()
                traceback.print_exc()
                print()
                self.send_error_message(f"Error while executing command: {e}")
            
            message = {}
            message["success"] = True
            message["command"] = client_input["command"]

            # Feedback
            for k, v in feedback.items():
                message[k] = v

            to_send = pickle.dumps(message)
            self.socket.send(to_send)

            if self.verbose:
                print("Sent a response.")


if __name__ == "__main__":
    robot = RobotServer(verbose=1, on_real=True)
    robot.run()
