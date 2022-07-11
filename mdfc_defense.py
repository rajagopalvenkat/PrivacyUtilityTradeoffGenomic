import numpy as np
import docplex.mp.model as cpx
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import time

#print("STDOUT check")

delta=1e-06

def D(f, n):
	return np.clip(np.power((1-f), (2*n), dtype=np.longdouble), 5e-307, 0.999999)

def A(f, n):
	return np.log(1-D(f,n), dtype=np.longdouble) - np.log(1-(delta*D(f,n-1)), dtype=np.longdouble)

def B(f,n):
	return np.log(D(f,n), dtype=np.longdouble) - np.log(delta*D(f, n-1), dtype=np.longdouble)

bits=pickle.load(open('In_Pop.pkl', 'rb'))

d=[]

for b in bits:
	d+=[b.tolist()]
d=np.array(d, dtype=int)

bits=[]

bits=pickle.load(open('Not_In_Pop.pkl', 'rb'))

d_n=[]

for b in bits:
	d_n+=[b.tolist()]
d_n=np.array(d_n, dtype=int)

bits=[]

x_beacon = np.array([int(x.split('\t')[0]) for x in open('In_Pop_Beacon.txt').readlines()])
aafs = np.array([float(x.rstrip('\n').split('\t')[0]) for x in open('AAFs').readlines()])

theta=float(sys.argv[1])
mask_wt=float(sys.argv[2])

flipped_dv=np.zeros(1338843, dtype=int)
masked_dv=np.zeros(1338843, dtype=int)
flipped = []
masked=[]
uncovered=np.ones(400, dtype=int)

LD_df = pd.read_csv('LD_keys_'+str(sys.argv[3]), names=['keys'], dtype=int)
val_list=[np.fromstring(l.rstrip(', \n'), sep=', ', dtype=int) for l in open('LD_vals_'+str(sys.argv[3])).readlines()]
LD_df['vals']=val_list
val_list=[]
LD_df['sizes']=[v.size for v in LD_df['vals']]
denom=np.ones(1338843, dtype=int)
for j in LD_df['keys'].unique():
	denom[j]=LD_df[LD_df['keys']==j]['sizes']

Delt_flip = d*x_beacon*(B(aafs, 400)-A(aafs, 400))
Delt_mask = d*x_beacon*(-A(aafs, 400))

eta_init=np.sum((d*x_beacon*A(aafs,400))+(d*(1-x_beacon)*B(aafs,400)), axis=1)

d[:,np.where(x_beacon==0)]=0
score_i=np.zeros(400)
prev_sum=400

sel_snps=np.where(x_beacon==1)[0]

flp_ind=[]
dvars=[]
fc=0
mc=0

def correl_attack():
	x_beacon_copy = np.copy(x_beacon)
	x_beacon_copy[flipped]=0
	flipped_copy = np.copy(np.array(flipped, dtype=int))
	masked_copy = np.copy(np.array(masked, dtype=int))

	for j in flipped_copy:
		if j in LD_df['keys'].unique():
			if np.mean(x_beacon_copy[np.array(LD_df[LD_df['keys']==j]['vals'])[0]])>=0.75:
				x_beacon_copy[j]=1
				np.delete(flipped_copy, np.where(flipped_copy==j))
	for j in masked_copy:
		if j in LD_df['keys'].unique():
			if np.mean(x_beacon_copy[np.array(LD_df[LD_df['keys']==j]['vals'])[0]])>=0.75:
				x_beacon_copy[j]=1
				np.delete(masked_copy, np.where(masked_copy==j))

	inds=np.delete(np.arange(1338843), masked_copy)
	eta=np.sum((d[:,inds]*x_beacon_copy[inds]*A(aafs[inds],400))+(d[:,inds]*(1-x_beacon_copy[inds])*B(aafs[inds],400)), axis=1)
	priv = np.sum(eta>theta)
	return 400-priv

def get_scores():
	#scores = np.append(np.mean(Delt_flip[uncovered==1], axis=0), mask_wt*np.mean(Delt_mask[uncovered==1], axis=0))
	scores = np.append(np.mean(Delt_flip[uncovered==1], axis=0)/denom, mask_wt*np.mean(Delt_mask[uncovered==1], axis=0)/denom)
	dec_vars = np.append(np.ones(int(len(scores)/2)), np.zeros(int(len(scores)/2)))
	indices = np.append(np.arange(1338843), np.arange(1338843))
	dec_vars_mod = np.array([y for (x,y,z) in sorted(zip(scores, dec_vars, indices), key=lambda pair: pair[0])])[::-1]
	indices_mod = np.array([z for (x,y,z) in sorted(zip(scores, dec_vars, indices), key=lambda pair: pair[0])])[::-1]
	return dec_vars_mod, indices_mod

f=open("flipcounts/mix_def_results/fix/mdfc_theta_"+str(theta)+'_w_'+str(mask_wt)+"_"+str(sys.argv[3]), 'w')

while uncovered.sum()>0:
	print(fc, mc, 400-uncovered.sum())
	dec_vars_mod, indices_mod = get_scores()
	
	for j,dv in zip(indices_mod, dec_vars_mod):
		if flipped_dv[j]==0 and masked_dv[j]==0:
			if dv==1:
				flipped_dv[j]=1
				fc+=1
				score_i+=Delt_flip[:,j]
				Delt_flip[:,j]=-float('inf')
				Delt_mask[:,j]=-float('inf')
				flp_ind+=[j, ]
				flipped+=[j,]
				x_beacon[j]=0
				dvars+=[dv, ]

				if j in LD_df['keys'].unique():
					print("Additional Flips")
					vs = np.array(LD_df[LD_df['keys']==j]['vals'])[0]
					asd=0
					print(j, len(vs))
					for v in vs:
						if flipped_dv[v]==0 and masked_dv[v]==0:
							print(asd)
							asd+=1
							flipped_dv[v]=1
							fc+=1
							score_i+=Delt_flip[:,v]
							Delt_flip[:,v]=-float('inf')
							Delt_mask[:,v]=-float('inf')
							flp_ind+=[v, ]
							flipped+=[v,]
							x_beacon[v]=0
							dvars+=[dv, ]

				uncovered[np.where(score_i+eta_init>=theta)]=0
				
				if uncovered.sum()<prev_sum:
					prev_sum=uncovered.sum()
					f.write(str(fc) + ", "+ str(mc) +", " +str(400-uncovered.sum()) +'\n')
					f.write("Correl Attack: " + str(correl_attack())+'\n')
					break
	
			if dv==0:
				masked_dv[j]=1
				mc+=1
				score_i+=Delt_mask[:,j]
				Delt_flip[:,j]=-float('inf')
				Delt_mask[:,j]=-float('inf')
				flp_ind+=[j, ]
				masked+=[j,]
				dvars+=[dv, ]

				if j in LD_df['keys'].unique():
					vs = np.array(LD_df[LD_df['keys']==j]['vals'])[0]
					for v in vs:
						if flipped_dv[v]==0 and masked_dv[v]==0:
							masked_dv[v]=1
							mc+=1
							score_i+=Delt_mask[:,v]
							Delt_flip[:,v]=-float('inf')
							Delt_mask[:,v]=-float('inf')
							flp_ind+=[v, ]
							masked+=[v,]
							#x_beacon[v]=0
							dvars+=[dv, ]

				uncovered[np.where(score_i+eta_init>=theta)]=0
				
				if uncovered.sum()<prev_sum:
					prev_sum=uncovered.sum()
					f.write(str(fc) + ", "+ str(mc) +", " +str(400-uncovered.sum()) +'\n')
					f.write("Correl Attack: " + str(correl_attack())+'\n')
					break

f.write(str(fc) + ", "+ str(mc) +", " +str(400-uncovered.sum()) +'\n')
f.close()

dvf = open("dvar_mdfc_fix_"+str(theta)+"_"+str(mask_wt)+"_"+str(sys.argv[3]), 'w')

for i,d in zip(flp_ind, dvars):
	dvf.write(str(i) + ", " + str(d) + "\n")
dvf.close()




























