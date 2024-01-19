class Resolution:
    def __init__(self, width, height):
        self._width = width
        self._height = height
        
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
        

class Vector2D:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
        