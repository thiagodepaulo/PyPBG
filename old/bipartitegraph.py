
class BipartiteGraph:

	def __init__(self):
		self.typea = dict()
		self.typeb = dict()
		self.weight = dict()

	def add_edge(self, a, b, w_ab):
		self.typea.setdefault(a,[]).append(b)
		self.typeb.setdefault(b,[]).append(a)
		self.weight[(a,b)] = w_ab

	def add_weight(self, e, w):
		if e in self.weight:
			self.weight[e] = w

	def a_len(self):
		return len(self.typea)

	def b_len(self):
		return len(self.typeb)

	def edges_len(self):
		return len(self.weight)

	def a_neig(self,a):
		if a in self.typea:
			return self.typea[a]
		else:
			return []

	def b_neig(self,b):
		if b in self.typeb:
			return self.typeb[b]
		else:
			return []

	def w(self, a, b):
		return self.weight[(a,b)]

	def w_a_neig(self,a):
		return [(b, self.w(a,b)) for b in self.a_neig(a)]

	def w_b_neig(self,b):
		return [(a, self.w(a,b)) for a in self.b_neig(b)]

	def a_vertices(self):
		return self.typea.keys()

	def b_vertices(self):
		return self.typeb.keys()

	def __repr__(self):
		return str(self.typea)

	def edges(self):
		l = []
		for a in self.a_vertices():
			for b, w in self.w_a_neig(a):
				l.append((a, b, w))			
		return l

	def load(self,arq):
		with open(arq,'r') as fin:
			for line in fin:
				a, b, w = line.split()
				self.add_edge(int(a), int(b), float(w))

	def save(self, arq):
		with open(arq,'w') as fout:
			for a, b, w in self.edges():
				fout.write(str(a)+' '+str(b)+' '+str(w)+'\n')
