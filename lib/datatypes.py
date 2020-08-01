#!/usr/bin/env python
# coding: utf8

class Dot:
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

	def __iter__(self):
		return iter((self.x, self.y))

	def __repr__(self):
		return f'({self.x}, {self.y})'

	def __eq__(self, vec):
		return self.x == vec.x and self.y == vec.y

	def __hash__(self):
		return hash((self.x, self.y))

class Vector(Dot):
	def __init__(self, x=0, y=0):
		super().__init__(x, y)

	def __neg__(self):
		return Vector(-self.x, -self.y)

	def __add__(self, vec):
		return Vector(self.x + vec.x, self.y + vec.y)

	def __sub__(self, vec):
		return Vector(self.x - vec.x, self.y - vec.y)

	def __mul__(self, n):
		return Vector(self.x * n, self.y * n)

	def __rmul__(self, n):
		return Vector(self.x * n, self.y * n)

	def __truediv__(self, n):
		return Vector(self.x / n, self.y / n)

	def __iadd__(self, vec):
		self.x += vec.x
		self.y += vec.y
		return self

	def __isub__(self, vec):
		self.x -= vec.x
		self.y -= vec.y
		return self

	def __imul__(self, n):
		self.x *= n
		self.y *= n
		return self

	def __itruediv__(self, n):
		self.x /= n
		self.y /= n
		return self

	@property
	def T(self):
		return Vector(self.y, self.x)

	# the dot product: self \cdot vec
	def pd_dot(self, vec):
		return self.x * vec.x + self.y * vec.y

	# the cross product: self \times vec
	def pd_cross(self, vec):
		return self.x * vec.y - self.y * vec.x

	def trans_linear(self, matrix, inplace=True):
		# matrix: TransMatrix
		new_x = matrix[0][0] * self.x + matrix[1][0] * self.y
		new_y = matrix[0][1] * self.x + matrix[1][1] * self.y
		if inplace:
			self.x, self.y = (new_x, new_y)
			return self
		else:
			return Vector(new_x, new_y)

class Rect:
	def __init__(self, x0, y0, x1, y1):
		self.x0, self.y0 = x0, y0
		self.x1, self.y1 = x1, y1

	def __iter__(self):
		return iter((self.x0, self.y0, self.x1, self.y1))

	def is_inside(self, node):
		return node.x >= self.x0 and node.x <= self.x1 \
				and node.y >= self.y0 and node.y <= self.y1

	def extend(self, node):
		if node.x < self.x0:
			self.x0 = node.x
		elif node.x > self.x1:
			self.x1 = node.x

		if node.y < self.y0:
			self.y0 = node.y
		elif node.y > self.y1:
			self.y1 = node.y

class Color_Grad:
	def __init__(self, n=None):
		self._n = n

	def __getitem__(self, index):
		if index < 0 or index >= self._n:
			raise IndexError('color index out of range')
		else:
			ix = index/self._n

		if   ix < 1/6:	x = 6*ix-0; return (0,1,x,1)
		elif ix < 2/6:	x = 2-6*ix; return (0,x,1,1)
		elif ix < 3/6:	x = 6*ix-2; return (x,0,1,1)
		elif ix < 4/6:	x = 4-6*ix; return (1,0,x,1)
		elif ix < 5/6:	x = 6*ix-4; return (1,x,0,1)
		else:			x = 6-6*ix; return (x,1,0,1)

# vi: set ts=4 noexpandtab foldmethod=indent :
