
import numpy as np
import math
import time
from bipartitegraph import BipartiteGraph

class BGP(object):

	def __init__(self, _G, _alpha, _beta, _lmaxitr, _gmaxitr, _local_threshold = 1e-6, _global_threshold = 1e-6, max_time=18000):
		self.G = _G
		self.beta = _beta
		self.alpha = _alpha
		self.GLOBAL_MAX_ITR = _gmaxitr
		self.LOCAL_MAX_ITR = _lmaxitr
		self.LOCAL_CONV_THRESHOLD = _local_threshold
		self.GLOBAL_CONV_THRESHOLD = _global_threshold	
		self.save = False	
		self.dir = ""		
		self.outA = "A"
		self.outB = "B"
		self.max_time = max_time	# seconds

	def create_rand_maps(self, K):
		self.K = K
		A = self.create_rand_matrix_A(K)
		B = self.create_rand_matrix_B(K)
		Amap = dict()
		Bmap = dict()
		for j, d_j in enumerate(self.G.a_vertices()):
			Amap[d_j] = A[j]
		for i, w_i in enumerate(self.G.b_vertices()):
			Bmap[w_i] = B[i]
		return Amap, Bmap	
	
	
	def create_rand_matrices(self, K):
		self.K = K
		return (self.create_rand_matrix_A(K), self.create_rand_matrix_B(K))

	def create_rand_matrix_B(self, K):
		N = self.G.b_len() 	# number of words
		return np.random.dirichlet(np.ones(N), K).transpose() 	# B (N x K) matrix

	def create_rand_matrix_A(self, K):
		M = self.G.a_len()	# number of documents
		return np.random.dirichlet(np.ones(K), M)	# A (M x K) matrix

	def create_ones_matrix_A(self, K):
		M = self.G.a_len()	# number of documents
		return np.ones(shape=(M,K))		
	
	def create_fromB_matrix_A(self, B):
		K = len(B[0]) 
		M = self.G.a_len()	# number of documents
		A = np.zeros(shape=(M,K))
		for d_j in self.G.a_vertices():
			for w_i, f_ji in self.G.w_a_neig(d_j):
				A[d_j] += f_ji * B[w_i]
				
		return A

	def create_fromA_matrix_B(self, A):
		K = len(A[0])
		N = self.G.b_len() 	# number of words
		B = np.zeros(shape=(N,K))
		for w_i in self.G.b_vertices():
			for d_j, f_ji in self.G.w_b_neig(w_i):
				B[w_i] += f_ji * A[d_j]
		return self.normalizebycolumn(B)

	def normalizedbycolumn_map(self, B):	
		n = len(B.values()[0])
		col_sum = np.zeros(n)
		for key in B:
			vet = B[key]
			for i in xrange(n):
				col_sum[i] += vet[i]
		for key in B:
			vet = B[key]
			for i in xrange(n):
				vet[i] /= col_sum[i]
				vet[i] = self.beta + vet[i]
		return B

	def normalizebycolumn_plus_beta(self, B):
		if isinstance(B, dict):
			return self.normalizedbycolumn_map(B)
		nrow, ncol = B.shape
		for i in xrange(ncol):
			B[:,i] /= B[:,i].sum()
		return self.beta + B

	def global_propag(self, A, B):	
		for w_i in self.G.b_vertices():
			nB_i = np.zeros(self.K)
			for d_j, f_ji in self.G.w_b_neig(w_i):
				H = (A[d_j] * B[w_i] )
				nB_i += f_ji * (H / H.sum())
			B[w_i] = nB_i
		# B = self.beta + self.normalizebycolumn(B)
		return self.normalizebycolumn_plus_beta(B)

	def Q2(self, A, B, alpha):
		CONST = 0.0000001
		_sum = 0
		for d_j in self.G.a_vertices():			
			for w_i, f_ji in self.G.w_a_neig(d_j):
				AB_ji = A[d_j]*B[w_i]
				C_ji = (AB_ji / AB_ji.sum())
				_sum += sum((f_ji * C_ji) * (np.log((AB_ji + CONST) / (C_ji + CONST))))								
			_sum -= sum((alpha - A[d_j]) * np.log(A[d_j] + CONST) - A[d_j]*(np.log(A[d_j]  + CONST ) - 1))
		return _sum

	def Q(self, A, B):
		_sum = 0
		for d_j in self.G.a_vertices():
			for w_i, f_ji in self.G.w_a_neig(d_j):
				sumAjBi = sum(A[d_j]*B[w_i])
				_sum += f_ji * np.log( f_ji / (sumAjBi)) - f_ji + (sumAjBi)
		return _sum

	def local_propag(self, d_j, A_j, B): 
		nA_j = np.zeros(len(A_j))
		for w_i, f_ji in self.G.w_a_neig(d_j):
			H = (A_j * B[w_i]) 
			nA_j += f_ji * (H / H.sum())
		nA_j += self.alpha
		return nA_j


	def bgp(self, A, B):
		oldq = float('infinity')
		global_niter = 0
		
		self.K = len(A.values()[0]) if isinstance(A, dict) else len(A[0])

		t0 = time.time()
		while global_niter <= self.GLOBAL_MAX_ITR :
			global_niter += 1

			if time.time() - t0 > self.max_time:
				break
			if self.save :
				np.save(self.dir+'/'+self.outA+'_'+str(time.time() - t0), A)
				#np.save(self.dir+'/'+self.outB+str(global_niter), B)

			for d_j in self.G.a_vertices():
				local_niter = 0
				if d_j % 1000 == 0: print d_j
				while local_niter <= self.LOCAL_MAX_ITR :
					local_niter += 1
					oldA_j = np.array(A[d_j])
					A[d_j] = self.local_propag(d_j, A[d_j], B)
					mean_change = np.mean(abs(A[d_j] - oldA_j))
					if mean_change <= self.LOCAL_CONV_THRESHOLD: 
						#print 'convergiu itr %s' %local_niter
						break
			self.global_propag(A, B)
			q = self.Q2(A,B,self.alpha)
			print 'itr %s Q %s' % (global_niter, q)
			#if abs(q - oldq) <= self.GLOBAL_CONV_THRESHOLD:
			#	print '\t\t **GLOBAL convergiu em %s iteracoes' %global_niter
			#	break
			#oldq = q
			
	
#	def global_propag2(self, A, B):	
#		for w_i in self.G.b_vertices():
#			C = dict()
#			for d_j in self.G.b_neig(w_i):
#				H = (A[d_j] * B[w_i] )
#				C[d_j] = H / H.sum()				
#			B[w_i] = np.array([f_ji*C[d_j] for d_j, f_ji in self.G.w_b_neig(w_i)]).sum(axis=0)
#		B = self.beta + self.normalizebycolumn(B)
#		return B


#	def local_propag2(self, d_j, A_j, B):
#		C = dict()			
#		for w_i in self.G.a_neig(d_j):
#			H = (A_j * B[w_i]) 
#			C[w_i] = H / H.sum()
#		A_j = self.alpha + np.array([f_ji*C[w_i] for w_i, f_ji in self.G.w_a_neig(d_j)]).sum(axis=0)
#		#A_j = A_j / A_j.sum()
#		return A_j


if __name__ == '__main__':
	import sys 
	import os

	if len(sys.argv) != 8:
		print "run: python %s arq_graph K alpha beta local_niter global_niter out_dir" %sys.argv[0]
		sys.exit(0)
	arq_graph = sys.argv[1]
	K = int(sys.argv[2])
	alpha = float(sys.argv[3])
	beta = float(sys.argv[4])
	local_niter = int(sys.argv[5])
	global_niter = int(sys.argv[6])
	out = sys.argv[7]
	# _a, _alpha, _gmaxitr, _lmaxitr

	g = BipartiteGraph()
	print 'carregando grafo...'
	g.load(arq_graph)
	print 'pronto!'

	bgp = BGP(g, alpha, beta, alpha, local_niter, global_niter)	
	A, B = bgp.create_rand_maps(K)
	bgp.bgp(A,B)
	
	np.save(out+'/A', A)
	print A
	np.save(out+'/B', B)
	print B

