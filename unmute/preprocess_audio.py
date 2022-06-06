import numpy as np
import audio_tools as aud
import audio_read as ar

SR = 8016
FPS = 24
SPF = int(SR/FPS)
NFRAMES = 9 # size of input volume of frames
MARGIN = int(NFRAMES/2)
OVERLAP = 1.0/2
LPC_ORDER = 8
NET_OUT = int(1/OVERLAP)*(LPC_ORDER+1)

def process_audio(af):
	# audio processing
	(y,sr) = ar.audio_read(af,sr=SR)
	win_length = SPF
	hop_length = int(SPF*OVERLAP)
	[a,g,e] = aud.lpc_analysis(y,LPC_ORDER,window_step=hop_length,window_size=win_length)
	lsf = aud.lpc_to_lsf(a)
	if len(lsf) % 2 != 0:
		lsf = lsf[:-1]
		g = g[:-1]
	lsf_concat = np.concatenate((lsf[::2,:],lsf[1::2,:]),axis=1) # MAGIC NUMBERS for half overlap
	g_concat = np.concatenate((g[::2,:],g[1::2,:]),axis=1) # MAGIC NUMBERS for half overlap
	feat = np.concatenate((lsf_concat,g_concat),axis=1)
	return feat
