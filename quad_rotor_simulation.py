import os
import time
import datetime
import numpy as np
import cv2
from multiprocessing import Process

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile, AmbientLight, DirectionalLight, VBase3, VBase4, Vec4, Vec2, \
    KeyboardButton, LPoint3f, NodePath
from panda3d.physics import ActorNode, ForceNode, LinearVectorForce, AngularVectorForce

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = 'QuadRotorModels/'
quad_model_filename = 'quad'

loadPrcFile('./config/conf_eval.prc')

i = LPoint3f(1, 0, 0)
j = LPoint3f(0, 1, 0)
k = LPoint3f(0, 0, 1)


def sign(x):
    return 1 if x > 0 else -1


class QuadRotorSimulation(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.scene = self.loader.loadModel(models_dir + "hallway.bam")  # Load the environment model
        self.scene.reparentTo(self.render)
        self.scene.setScale(1, 1, 1)
        self.scene.setPos(0, 0, 1)
        self.scene.setHpr(90, 0, 0)

        # Add an ambient light and set sky color
        sky_col = VBase3(135 / 255.0, 206 / 255.0, 235 / 255.0)
        self.set_background_color(sky_col)
        alight = AmbientLight("sky")
        alight.set_color(VBase4(sky_col * 0.04, 1))
        alight_path = self.render.attachNewNode(alight)
        self.render.set_light(alight_path)

        # # 4 perpendicular lights (flood light)
        for light_no in range(4):
            d_light = DirectionalLight('directionalLight')
            d_light.setColor(Vec4(*([0.3] * 4)))
            d_light_NP = self.render.attachNewNode(d_light)
            d_light_NP.setHpr(-90 * light_no, 0, 0)
            self.render.setLight(d_light_NP)

        # # 1 directional light (Sun)
        sun_light = DirectionalLight('directionalLight')
        sun_light.setColor(Vec4(*([0.7] * 4)))  # directional light is dim green
        sun_light.getLens().setFilmSize(Vec2(0.8, 0.8))
        sun_light.getLens().setNearFar(-0.3, 12)
        sun_light.setShadowCaster(True, 2 ** 7, 2 ** 7)
        self.dlightNP = self.render.attachNewNode(sun_light)
        self.dlightNP.setHpr(0, -65, 0)

        # Turning shader and lights on
        self.render.setLight(self.dlightNP)

        # Load and transform the quadrotor actor.
        self.quad_model = self.loader.loadModel(models_dir + f'{quad_model_filename}.egg')
        self.prop_models = []

        for prop_no in range(4):
            prop = self.loader.loadModel(models_dir + 'propeller.egg')
            x = 0 if prop_no % 2 == 1 else (-0.26 if prop_no == 0 else 0.26)
            y = 0 if prop_no % 2 == 0 else (-0.26 if prop_no == 3 else 0.26)
            prop.setPos(x, y, 0)
            prop.reparentTo(self.quad_model)
            self.prop_models.append(prop)

        self.prop_models = tuple(self.prop_models)
        # self.quad_model.reparentTo(self.scene)
        self.quad_model.setPos(0, 0, 2)
        self.quad_neutral_hpr = (90, 0, 0)
        self.quad_model.setHpr(*self.quad_neutral_hpr)

        # env cam
        self.cam_neutral_pos = (0, -4, 3)
        self.cam.reparentTo(self.scene)
        # self.cam_neutral_pos = (-4, 0, 1)
        # self.cam.reparentTo(self.quad_model)

        self.cam.setPos(*self.cam_neutral_pos)
        self.cam.lookAt(self.quad_model)

        self.enableParticles()
        node = NodePath("PhysicsNode")
        node.reparentTo(self.scene)
        self.actor_node = ActorNode("quadrotor-physics")
        # self.actor_node.getPhysicsObject().setMass(1)
        self.actor_node_physics = node.attachNewNode(self.actor_node)
        self.physicsMgr.attachPhysicalNode(self.actor_node)
        self.quad_model.reparentTo(self.actor_node_physics)

        # add gravity
        # gravity_force_node = ForceNode('world-forces')
        # gravityForce = LinearVectorForce(0, 0, -0.1)  # gravity acceleration
        # gravity_force_node.addForce(gravityForce)
        # self.physicsMgr.addLinearForce(gravityForce)

        self.time = datetime.datetime.today().strftime('%Y-%m-%d-%H.%M.%S')
        self.simulation_folder = "\sims\\" + self.time + '\\'
        self.simulation_folder_path = ROOT_DIR + self.simulation_folder
        os.makedirs(self.simulation_folder_path)
        self.movements = ''

        self.taskMgr.add(self.camera_move, 'Camera Movement')
        self.taskMgr.add(self.quad_move, 'Quad Movement')
        self.taskMgr.add(self.rotate_propellers, 'Propellers Rotation')
        self.taskMgr.add(self.save_image, 'Screenshot Capture')

        # self.buffer: GraphicsBuffer = self.win.makeTextureBuffer(name='buffer', x_size=84, y_size=84, tex=None, to_ram=True)
        # self.buffer.setActive(1)
        self.images = []
        self.image_index = 1

    def camera_move(self, task):
        """
        Moves the camera about the quadcopter
        """

        keys_vs_moves = {'k': i,
                         'h': -i,
                         'i': j,
                         'y': -j,
                         'u': k,
                         'j': -k
                         }

        mat = np.array(self.cam.getMat())[0:3, 0:3]
        move_total = LPoint3f(0, 0, 0)

        for key, move in keys_vs_moves.items():
            pressed_key = self.mouseWatcherNode.is_button_down(KeyboardButton.asciiKey(key))
            if pressed_key:
                move = LPoint3f(move)
                move_total += move

        # if any([abs(coordinate) > 0 for coordinate in move_total]):
        # ROTATE COORDINATE SYSTEM (TO CAMERA)
        move_total = LPoint3f(*tuple(np.dot(mat.T, np.array(move_total))))
        proportionality_constant = 0.05
        cam_pos = self.cam.getPos() + move_total * proportionality_constant
        self.cam.setPos(cam_pos)
        self.cam.lookAt(self.quad_model)

        return task.cont

    def quad_move(self, task):
        position_proportionality_constant = 0.0007
        angle_proportionality_constant = 0.2
        angle_max = 10
        angle_directions = {i: k, j: -j}

        keys_vs_moves = {'w': i,
                         's': -i,
                         'a': j,
                         'd': -j,
                         'e': k,
                         'q': -k}

        move_total = LPoint3f(0, 0, 0)
        angle_total = LPoint3f(0, 0, 0)

        for key, move in keys_vs_moves.items():
            pressed_key = self.mouseWatcherNode.is_button_down(KeyboardButton.asciiKey(key))
            if pressed_key:
                move_total += move

                if key not in ['e', 'q', 'm']:
                    angle_total += LPoint3f(*tuple(sign(move) *
                                                   np.array(
                                                       angle_directions[LPoint3f(*tuple(int(abs(c)) for c in move))])
                                                   * angle_proportionality_constant))
                else:
                    prop_sign = 1 if key == 'e' else -1

                    for prop in self.prop_models:
                        prop.setHpr(prop.get_hpr() + LPoint3f(10 * prop_sign, 0, 0))

                self.movements += key

                with open(ROOT_DIR + self.simulation_folder + 'movements.txt', 'w') as file:
                    file.write(self.movements)

                    # self.movie(namePrefix=self.simulation_folder, duration=1.0, fps=30,
                    #       format='png', sd=4, source=None)

        if any([abs(coordinate) > 0 for coordinate in angle_total]):
            # tilt_force_node = ForceNode('tilt-force')
            # tilt_force = AngularVectorForce(*angle_total)
            # tilt_force_node.addForce(tilt_force)
            #
            # self.actor_node.getPhysical(0).addAngularForce(tilt_force)

            desired_quad_hpr = list(self.quad_model.getHpr() + angle_total)

            for index, coordinate in enumerate(desired_quad_hpr):
                if coordinate:
                    desired_quad_hpr[index] = sign(coordinate) * min(abs(coordinate), 90 + angle_max
                    if index == 0 else angle_max)

            desired_quad_hpr = LPoint3f(*desired_quad_hpr)
            self.quad_model.setHpr(desired_quad_hpr)

        if any([abs(coordinate) > 0 for coordinate in move_total]):

            movement_force_node = ForceNode('movement-force')
            movement_force = LinearVectorForce(*(- move_total * position_proportionality_constant))
            # movement_force.setMassDependent(1)
            movement_force_node.addForce(movement_force)

            self.actor_node.getPhysical(0).addLinearForce(movement_force)

            # time.sleep(0.1)

            # thruster.setP(-45)  # bend the thruster nozzle out at 45 degrees

            # desired_quad_pos = self.quad_model.getPos() + move_total * position_proportionality_constant
            # self.quad_model.setPos(desired_quad_pos)

        return task.cont

    def rotate_propellers(self, task):
        for prop in self.prop_models:
            prop.setHpr(prop.get_hpr() + LPoint3f(30, 0, 0))

        return task.cont

    # def produce_video(self):

    def save_image(self, task):
        self.image_index += 1
        if self.image_index % 10 == 0:
            self.screenshot(namePrefix=rf'{self.simulation_folder}\image')

        if self.image_index % 100 == 0:
            # p = Process(target=my_function, args=(queue, 1))
            # p.start()

            video_name = rf'{self.simulation_folder_path}\video.avi'

            images = [img for img in os.listdir(self.simulation_folder_path) if img.endswith(".jpg")]
            frame = cv2.imread(os.path.join(self.simulation_folder_path, images[0]))
            height, width, layers = frame.shape

            video = cv2.VideoWriter(video_name, 0, 30, (width, height))

            for image in images:
                video.write(cv2.imread(os.path.join(self.simulation_folder_path, image)))

            cv2.destroyAllWindows()
            video.release()

        return task.cont


app = QuadRotorSimulation()
app.run()
