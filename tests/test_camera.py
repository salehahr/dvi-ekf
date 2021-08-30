import unittest

from context import Camera, Config

config  = Config()

class TestCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = Camera(filepath='./trajs/mandala0_gt.txt', max_vals=20)
        cls.cam_interp = cls.cam.interpolate(interframe_vals = 10)

    @unittest.skip('Skip plot')
    def test_plot(self):
        self.cam.plot(config)
        self.cam_interp.plot(config)

    def test_plot_notch(self):
        self.cam.plot_notch(config)

if __name__ == '__main__':
    from functions import run_only
    run_only(TestCamera)
