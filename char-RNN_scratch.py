#-----------------------------------IMPORTING THE ONLY LIBRARY NEEDED WHICH IS NUMPY---------------------------------------#
import numpy as np
#--------------------------------DATA I/O FROM A TEXT FILE WHICH IS ALREADY CREATED AND LOADED WITH RANDOM TEXT------------#

data=open("input.txt",'r').read()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)
print("The text data has {} characters and {} unique characters".format(data_size,vocab_size))
char_to_ix={ ch:i for i,ch in enumerate(chars) }
ix_to_chars={ i:ch for i,ch in enumerate(chars) }

#-------------------------------------DEFINING OUR MODEL HYPERPARAMETERS----------------------------------------------------#
hidden_size=100
seq_length=25
learning_rate=1e-1

#------------------------------------DEFINING OUR MODEL PARAMETERS----------------------------------------------------------#

Wxh=np.random.randn(hidden_size,vocab_size)*0.01 # initialization of weights matrix corresponding to input-hidden layer mapping
Whh=np.random.randn(hidden_size,hidden_size)*0.01 # initialization of weights matrix corresponding to hidden-hidden layer mapping
Why=np.random.randn(vocab_size,hidden_size)*0.01 # initilization of weights matrix corresponding to hidden-output layer mapping
bh=np.zeros((hidden_size,1)) # Bias for the hidden layer
by=np.zeros((vocab_size,1)) # Bias for the output layer

#----------------------------------DEFINING LOSS FUNCTION FOR FORWARD AND BACK PROPAGATION----------------------------------#
def lossFun(inputs,targets,hprev):
	xs,ys,hs,ps={},{},{},{}
	hs[-1]=np.copy(hprev)
	loss=0

	# Forward Pass
	for t in range(len(inputs)):
		xs[t]=np.zeros((vocab_size,1))
		xs[t][inputs[t]]=1
		hs[t]=np.tanh(np.dot(Wxh,xs[t])+np.dot(Whh,hs[t-1]) + bh) # hidden state values
		ys[t]=np.dot(Why,hs[t])+by # output state values
		ps[t]=np.exp(ys[t])/np.sum(np.exp(ys[t])) # Generating probabilities from softmax function
		loss+=-np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

	# Backward Pass
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
		np.clip(dparam,-5,5,out=dparam) # Gradient clipping for exploding gradients
		return loss,dWxh,dWhh,dWhy,dbh,dby,hs[len(inputs)-1]

#------------------DEFINING SAMPLE FUNCTION FOR SAMPLING FROM THE PROBABILITIES GENERATED FROM SOFTMAX FUNCTION-------------#

def sample(h,seed_ix,n):
	x=np.zeros((vocab_size,1))
	x[seed_ix]=1 #Initializing x vector corresponding to the first character of the text to be predicted
	ixes=[]
	for t in range(n):
		h=np.tanh(np.dot(Wxh,x)+np.dot(Whh,h)+bh)   # single forward prop
		y=np.dot(Why,h) + by  # calculating output
		p=np.exp(y)/np.sum(np.exp(y))  # generating probabilities from output using softmax function
		ix=np.random.choice(range(vocab_size),p=p.ravel()) # random sampling from the probability distribution
		x=np.zeros((vocab_size,1))
		x[ix]=1 # updating the next x vector with the sampled character's index
		ixes.append(ix)
	return ixes

#---------------------------------------------TRAINING OUR MODEL------------------------------------------------------------#

n,p=0,0
mWxh,mWhy,mWhh=np.zeros_like(Wxh),np.zeros_like(Why),np.zeros_like(Whh)
mbh,mby=np.zeros_like(bh),np.zeros_like(by) # Memory variables for Adagrad Optimizer
smooth_loss=-np.log(1.0/vocab_size)*seq_length # Initial loss at iteration 0

while True:
	if p+seq_length+1>=len(data) or n==0:
		hprev=np.zeros((hidden_size,1)) # Starting hprev -- resetting RNN memory
		p=0
	inputs=[char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets=[char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
	
	# SAMPLING AFTER EVERY 200 EPOCHS FOR DISPLAYING
	if n%200==0:
		sample_ix=sample(hprev,inputs[0],data_size-1)
		txt=''.join(ix_to_chars[ix] for ix in sample_ix)
		print('----\n {} \n----'.format(txt, ))

	loss,dWxh,dWhh,dWhy,dbh,dby,hprev =lossFun(inputs,targets,hprev)
	smooth_loss=smooth_loss*0.999 +loss*0.001

	# PRINTING TRAINING STATUS AFTER EVERY 200 EPOCHS
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
		
#-----------GENERATING THE INPUT TEXT USING OUR TRAINED RNN MODEL BY PROVIDING ONLY THE FIRST CHARACTER OF THE TEXT---------#	
starting_char=data[0]
hprev=np.zeros((hidden_size,1))
sample_ix=sample(hprev,char_to_ix[starting_char],data_size-1)
txt=''.join(ix_to_chars[ix] for ix in sample_ix)
print('----\n',starting_char,'{} \n----'.format(txt, ))








