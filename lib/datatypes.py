#!/usr/bin/env python
# coding: utf8

class Dot:
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

	def __iter__(self):
		return iter((self.x, self.y))

	def __repr__(self):
		return '({}, {})'.format(self.x, self.y)

class Vector(Dot):
	def __init__(self, x=0, y=0):
		super().__init__(x, y)

	def __neg__(self):
		return Vector(-self.x, -self.y)

	def __eq__(self, vec):
		return self.x == vec.x and self.y == vec.y

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

# vi: set ts=4 noexpandtab foldmethod=indent :
