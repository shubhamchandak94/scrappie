import sys
import numpy as np
import scrappy
import scipy.stats as st
import subprocess
import os
import distance
import util
PATH_TO_CPP_EXEC = "/raid/nanopore/shubham/scrappie/shubham/viterbi_nanopore.out"
NUM_TRIALS = 100
SYN_SUB_PROB = 0.0
SYN_DEL_PROB = 0.0
SYN_INS_PROB = 0.0

# from deepsimulator (probably more realistic dwell times)
def rep_rvs(size,a):
    # array_1 = np.ones(int(size*0.075)).astype(int)
    # samples = st.alpha.rvs(3.3928495261646932, 
    #     -7.6451557771999035, 50.873948369526737, 
    #     size=(size-int(size*0.075))).astype(int)
    # samples = np.concatenate((samples, array_1), 0)
    # samples[samples<1] = 2
    # print(a)
    a = a*5
    array_1 = np.ones(int(size*(0.075-0.015*a))).astype(int)
    samples = st.alpha.rvs(3.3928495261646932+a, 
        -7.6451557771999035+(2*a), 50.873948369526737, 
        size=(size-int(size*(0.075-0.015*a)))).astype(int)
    samples = np.concatenate((samples, array_1), 0)
    samples[samples<1] = 2
    np.random.shuffle(samples)
    return samples

msg_len = 144
hamming_list = []
edit_list = []
correct_list = []

for _ in range(NUM_TRIALS):
    msg = ''.join(np.random.choice(['0','1'], msg_len))
    print(msg)
    rnd = str(np.random.randint(10000000))
    with open('tmp.'+rnd,'w') as f:
        f.write(msg)
    subprocess.run([PATH_TO_CPP_EXEC,'encode','tmp.'+rnd,'tmp.enc.'+rnd])

    infile_seq = 'tmp.enc.'+rnd
    # can handle fasta header, but also works without it
    f = open(infile_seq)
    seq = f.readline().rstrip('\n')
    if seq[0] == '>':
            seq = f.readline().rstrip('\n')
    f.close()
    len_seq = len(seq)
    print('Length of seq: ', len_seq)
    print(seq)
    os.remove('tmp.'+rnd)
    os.remove('tmp.enc.'+rnd)
    syn_seq = util.simulate_indelsubs(seq, sub_prob = SYN_SUB_PROB, del_prob = SYN_DEL_PROB, ins_prob = SYN_INS_PROB)
    print('Length of synthesized sequence', len(syn_seq))
    print(syn_seq)
    squiggle_array = scrappy.sequence_to_squiggle(syn_seq,rescale=True).data(as_numpy=True)
    raw_data = np.array([])

    # for dwell time use deepsimulator since the one provided by scrappie is way too good
    # scrappie dwell gives around 3-4% edit distance while ds dwell gives around 15%
    ds_alpha = 0.1 # 0.1 is default parameter in deepsim
    squiggle_array[:,0] = rep_rvs(squiggle_array.shape[0], ds_alpha)

    for squig in squiggle_array:
            mean = squig[1]
            stdv = squig[2]
            dwell = squig[0]
            raw_data = np.append(raw_data, np.random.laplace(mean, stdv/np.sqrt(2), int(round(dwell))))

    print('Length of raw signal: ', len(raw_data))

    decoded_msg = scrappy.basecall_raw_viterbi_conv(raw_data,PATH_TO_CPP_EXEC, msg_len)
    print(decoded_msg)
    hamming_list.append(distance.hamming(msg,decoded_msg))
    print('Hamming distance',hamming_list[-1])
    edit_list.append(distance.levenshtein(msg,decoded_msg))
    print('Edit distance',edit_list[-1])
    correct_list.append((hamming_list[-1] == 0))
print('Summary statistics:')
print('Number total:', NUM_TRIALS)
print('Number correct:', sum(correct_list))
print('Average bit error rate:', sum(hamming_list)/(msg_len*NUM_TRIALS))
