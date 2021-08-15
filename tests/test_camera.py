import unittest

from context import Camera

class TestCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = Camera(filepath='./trajs/mandala0_gt.txt', max_vals=10)
        cls.cam_interp = cls.cam.interpolate(interframe_vals = 10)

    @unittest.skip('Skip plot')
    def test_plot(self):
        ax = self.cam.traj.plot()
        ax = self.cam_interp.traj.plot(axes=ax)
        plt.show()

if __name__ == '__main__':
    from functions import run_only
    run_only(TestCamera)
