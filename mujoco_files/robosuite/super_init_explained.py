class Base(object):
    def __init__(self):
        print("Base init'ed")

class ChildA(Base):
    def __init__(self):
        print("ChildA init'ed")
        Base.__init__(self)

class ChildB(Base):
    def __init__(self):
        print("ChildB init'ed")
        super().__init__()


class UserDependency(Base):
    def __init__(self):
        print("UserDependency init'ed")
        Base.__init__(self)


class UserDependency2(Base):
    def __init__(self):
        print("UserDependency2 init'ed")
        super().__init__()


class UserA(ChildA, UserDependency, UserDependency2):
    def __init__(self):
        print("UserA init'ed")
        super().__init__()


class UserB(ChildB, UserDependency, UserDependency2):
    def __init__(self):
        print("UserB init'ed")
        print(self.dof)
        print(self.ros())
        super().__init__()

    def _load_model(self):
        super._load_model()
        print("asda")

    @property
    def dof(self):
        return 14

    def ros(self):
        return 20

class check(UserB):
    def __init__(self, **kwargs):
        print("balsdlas")
        super().__init__(**kwargs)


check()
