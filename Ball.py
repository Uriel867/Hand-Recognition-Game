class Ball:
    def __init__(self,x,y,radius,speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def setX(self,new_x):
        self.x = new_x

    def setY(self,new_y):
        self.y = new_y

    def check_boundaries(self,frame):
         return ((self.x - self.radius <= 0) or (self.x + self.radius >= frame.shape[1]) or
                 (self.y - self.radius <= 0) or (self.y + self.radius >= frame.shape[0]))

