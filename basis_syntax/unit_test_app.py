import unittest


class App1Test(unittest.TestCase):

    def setUp(self) -> None:  # ​Python 3 新特性：类型注解
        print("setup")

    def test_app(self):
        self.assertEqual("1", "1")


