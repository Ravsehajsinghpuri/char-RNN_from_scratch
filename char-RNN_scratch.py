import numpy as np
# Data I/O and some organization
data=open("input.txt",'r').read()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)
print("The text data has {} characters and {} unique characters".format(data_size,vocab_size))
char_to_ix={ ch:i for i,ch in enumerate(chars) }
ix_to_chars={ i:ch for i,ch in enumerate(chars) }

# Hyperparameters
hidden_size=100
seq_length=25
learning_rate=1e-1

# model parameters
Wxh=np.random.randn(hidden_size,vocab_size)*0.01 # initialization of weights matrix corresponding to input-hidden layer mapping
Whh=np.random.randn(hidden_size,hidden_size)*0.01 # initialization of weights matrix corresponding to hidden-hidden layer mapping
Why=np.random.randn(vocab_size,hidden_size)*0.01 # initilization of weights matrix corresponding to hidden-output layer mapping
bh=np.zeros((hidden_size,1)) # bias for the hidden layer
by=np.zeros((vocab_size,1)) # bias for the output layer

def lossFun(inputs,targets,hprev):
	xs,ys,hs,ps={},{},{},{}
	hs[-1]=np.copy(hprev)
	loss=0

	#forward Pass
	for t in range(len(inputs)):
		xs[t]=np.zeros((vocab_size,1))
		xs[t][inputs[t]]=1
		hs[t]=np.tanh(np.dot(Wxh,xs[t])+np.dot(Whh,hs[t-1]) + bh)
		ys[t]=np.dot(Why,hs[t])+by
		ps[t]=np.exp(ys[t])/np.sum(np.exp(ys[t])) # generating probabilities using softmax function
		loss+=-np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

	# backward Pass
	dWxh,dWhh,dWhy=np.zeros_like(Wxh),np.zeros_like(Whh),np.zeros_like(Why)
	dbh,dby=np.zeros_like(bh),np.zeros_like(by)
	dhnext=np.zeros_like(hs[0])
	for t in reversed(range(len(inputs))):
		dy=np.copy(ps[t])
		dy[targets[t]]-=1 # backprop into y
		dWhy+= np.dot(dy,hs[t].T)
		dby+= dy
		dh=np.dot(Why.T,dy) + dhnext # backprop into h
		dhraw=(1-hs[t]*hs[t])*dh # backprop through tanh Non-Linearity
		dbh+=dhraw
		dWxh+=np.dot(dhraw,xs[t].T)
		dWhh+=np.dot(dhraw,hs[t-1].T)
		dhnext=np.dot(Whh.T,dhraw)

	for dparam in [dWxh,dWhh,dWhy,dbh,dby]:
		np.clip(dparam,-5,5,out=dparam) # clipping exploding gradients
		return loss,dWxh,dWhh,dWhy,dbh,dby,hs[len(inputs)-1]


def sample(h,seed_ix,n):
	x=np.zeros((vocab_size,1))
	x[seed_ix]=1
	ixes=[]
	for t in range(n):
		h=np.tanh(np.dot(Wxh,x)+np.dot(Whh,h)+bh)
		y=np.dot(Why,h) + by
		p=np.exp(y)/np.sum(np.exp(y))
		ix=np.random.choice(range(vocab_size),p=p.ravel())
		x=np.zeros((vocab_size,1))
		x[ix]=1
		ixes.append(ix)
	return ixes


n,p=0,0
mWxh,mWhy,mWhh=np.zeros_like(Wxh),np.zeros_like(Why),np.zeros_like(Whh)
mbh,mby=np.zeros_like(bh),np.zeros_like(by) # memory variables for Adagrad
smooth_loss=-np.log(1.0/vocab_size)*seq_length # initial loss at iteration 0

while True:
	if p+seq_length+1>=len(data) or n==0:
		hprev=np.zeros((hidden_size,1)) # Starting hprev -- resetting RNN memory
		p=0
	inputs=[char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets=[char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

	if n%200==0:
		sample_ix=sample(hprev,inputs[0],data_size-1)
		txt=''.join(ix_to_chars[ix] for ix in sample_ix)
		#print('----\n {} \n----'.format(txt, ))

	loss,dWxh,dWhh,dWhy,dbh,dby,hprev =lossFun(inputs,targets,hprev)
	smooth_loss=smooth_loss*0.999 +loss*0.001

	if n%200 == 0: print("iter {}, loss: {} ".format(n, smooth_loss)) 
	if smooth_loss<0.01:
		break
	for param,dparam,mem in zip([Wxh,Whh,Why,bh,by],[dWxh,dWhh,dWhy,dbh,dby],[mWxh,mWhh,mWhy,mbh,mby]):
		mem+=dparam*dparam
		param+= -learning_rate*dparam/np.sqrt(mem+1e-8)

	p+=seq_length
	n+=1
	if smooth_loss<0.01:
		break
starting_char='-'
hprev=np.zeros((hidden_size,1))
sample_ix=sample(hprev,char_to_ix[starting_char],data_size-1)
txt=''.join(ix_to_chars[ix] for ix in sample_ix)
print('----\n',starting_char,'{} \n----'.format(txt, ))








