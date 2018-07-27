from __future__ import print_function
 
import numpy as np
import tensorflow as tf

import os
import pickle
import copy
import sys
import html

# from utils import TextLoader
from model import Model

from utils import eprint # print to stderr
from utils import get_args # a global variable and the function what fills it
# from utils import args
#  made these global so I could examine them in the debugger TAC
# chars = []
# vocab = []

# args = {}

# def main():
#     global args
#     args = get_args() # set global args for now
#     sample_main()

def main():
    global args
    args = get_args()
    #  made these global so I could examine them in the debugger TAC

    # Create the model from the saved arguments, in inference mode.
    print("Creating model...")
    args.saved_args.batch_size = args.beam_width
    net = Model(args.saved_args, True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Make tensorflow less verbose; filter out info (1+) and warnings (2+) but not errors (3).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with tf.Session(config=config) as sess:

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(net.save_variables_list())
        # Restore the saved variables, replacing the initialized values.
        print("Restoring weights...")
        saver.restore(sess, args.model_path)
        
        chatbot(net, sess, args.max_length, args.beam_width,
                args.relevance, args.temperature, args.topn, args)

def initial_state(net, sess):
    # Return freshly initialized model states.
    return sess.run(net.zero_state)

def forward_text(net, sess, states, relevance, prime_text=None):
    if prime_text is not None:
        for char in prime_text:
            if relevance > 0.:
                # Automatically forward the primary net.
                _, states[0] = net.forward_model(sess, states[0], ord(char))
                # If the token is newline, reset the mask net state; else, forward it.

                if char == '\n':
                    states[1] = initial_state(net, sess)
                else:
                    _, states[1] = net.forward_model(sess, states[1], ord(char))
            else:
                _, states = net.forward_model(sess, states, ord(char))
    return states

def sanitize_text(text): # Strip out characters that are not part of the net's vocabulary.
    # all characters are in our vocabulary TAC
    # so this function is not needed
    return text

def initial_state_with_relevance_masking(net, sess, relevance):
    if relevance <= 0.: return initial_state(net, sess)
    else: return [initial_state(net, sess), initial_state(net, sess)]

def possibly_escaped_char(raw_chars):
    if raw_chars[-1] == ';':
        for i, c in enumerate(reversed(raw_chars[:-1])):
            if c == ';' or i > 8:
                return raw_chars[-1]
            elif c == '&':
                escape_seq = "".join(raw_chars[-(i + 2):])
                new_seq = html.unescape(escape_seq)
                backspace_seq = "".join(['\b'] * (len(escape_seq)-1))
                diff_length = len(escape_seq) - len(new_seq) - 1
                return backspace_seq + new_seq + "".join([' '] * diff_length) + "".join(['\b'] * diff_length)
    return raw_chars[-1]

def chatbot(net, sess, max_length, beam_width, relevance, temperature, topn, args):
    states = initial_state_with_relevance_masking(net, sess, relevance)
    while True:
        user_input = input('\n> ')
        if user_input == 'quit' : break
        user_command_entered, reset, states, relevance, temperature, topn, beam_width = process_user_command(
            user_input, states, relevance, temperature, topn, beam_width)
        if reset: states = initial_state_with_relevance_masking(net, sess, relevance)
        if not user_command_entered:
            states = forward_text(net, sess, states, relevance, sanitize_text("> " + user_input + "\n>"))
            computer_response_generator = beam_search_generator(sess=sess, net=net,
                initial_state=copy.deepcopy(states), initial_sample=ord(' '),
                early_term_token=ord('\n'), beam_width=beam_width,args=args)
            parts = []
            out_chars = []
            print('\n>')
            for i, char_token in enumerate(computer_response_generator):
                # out_chars.append(chars[char_token])
                out_chars.append(chr(char_token))
                # parts.append(possibly_escaped_char(out_chars), end='', flush=True)
                print(possibly_escaped_char(out_chars), end='', flush=True)
                states = forward_text(net, sess, states, relevance, chr(char_token))
                if i >= max_length: break

            states = forward_text(net, sess, states, relevance, sanitize_text("\n> "))
            # user_input = ''.join(parts)
            # print('\n>{}'.format(user_input))

def process_user_command(user_input, states, relevance, temperature, topn, beam_width):
    user_command_entered = False
    reset = False
    try:
        if user_input.startswith('--temperature '):
            user_command_entered = True
            temperature = max(0.001, float(user_input[len('--temperature '):]))
            print("[Temperature set to {}]".format(temperature))
        elif user_input.startswith('--relevance '):
            user_command_entered = True
            new_relevance = float(user_input[len('--relevance '):])
            if relevance <= 0. and new_relevance > 0.:
                states = [states, copy.deepcopy(states)]
            elif relevance > 0. and new_relevance <= 0.:
                states = states[0]
            relevance = new_relevance
            print("[Relevance disabled]" if relevance <= 0. else "[Relevance set to {}]".format(relevance))
        elif user_input.startswith('--topn '):
            user_command_entered = True
            topn = int(user_input[len('--topn '):])
            print("[Top-n filtering disabled]" if topn <= 0 else "[Top-n filtering set to {}]".format(topn))
        elif user_input.startswith('--beam_width '):
            user_command_entered = True
            beam_width = max(1, int(user_input[len('--beam_width '):]))
            print("[Beam width set to {}]".format(beam_width))
        elif user_input.startswith('--reset'):
            user_command_entered = True
            reset = True
            print("[Model state reset]")
    except ValueError:
        print("[Value error with provided argument.]")
    return user_command_entered, reset, states, relevance, temperature, topn, beam_width

# the length for which all beam_outputs are the same ???
# true if it is up to the early termination token ???
def consensus_length(beam_outputs, early_term_token):
    length = len(beam_outputs[0])
    for l in range(length):
        if l > 0 and beam_outputs[0][l-1] == early_term_token:
            return l-1, True
        for b in beam_outputs[1:]:
            if beam_outputs[0][l] != b[l]: return l, False
    return l, False

def scale_prediction(prediction, temperature):
    if (temperature == 1.0): return prediction # Temperature 1.0 makes no change
    np.seterr(divide='ignore')
    scaled_prediction = np.log(prediction) / temperature
    scaled_prediction = scaled_prediction - np.logaddexp.reduce(scaled_prediction)
    scaled_prediction = np.exp(scaled_prediction)
    np.seterr(divide='warn')
    return scaled_prediction

def NET_Probability(sess, net, states, input_sample, args):
    ''' pass it forward and get network probabilities '''
    prob, states = net.forward_model(sess, states, input_sample)
    print (np.shape(states))
    exit()
    return (prob,states)

def MMI_Probability(sess, net, states, input_sample, args):

    print (np.shape(states))
    exit()
    # eprint('MMI')
    # states should be a 2-length list: [primary net state, mask net state].
    # if we are doing relevance

    # Reset the mask probs when reaching mask_reset_token (newline).
    if input_sample == args.mask_reset_token: states[1] = initial_state(net, sess)

    primary_prob, states[0] = net.forward_model(sess, states[0], input_sample)
    mask_prob,    states[1] = net.forward_model(sess, states[1], input_sample)

    prob = np.exp(np.log(primary_prob) - args.relevance * np.log(mask_prob))

    prob[args.forbidden_token] = 0

    return (prob/sum(prob), states)

def Norm_MMI_Probability(sess, net, states, input_sample, args):

    # eprint('NORM')
    # states should be a 2-length list: [primary net state, mask net state].
    # if we are doing relevance

    # Reset the mask probs when reaching mask_reset_token (newline).
    if input_sample == args.mask_reset_token: states[1] = initial_state(net, sess)

    primary_prob, states[0] = net.forward_model(sess, states[0], input_sample)
    mask_prob,    states[1] = net.forward_model(sess, states[1], input_sample)

    prob = np.exp(np.log(primary_prob) - args.relevance * np.log(mask_prob))

    # calculate entropies for the 
    Hp = sum([-i*np.log(i) for i in primary_prob])
    Hm = sum([-i*np.log(i) for i in mask_prob])

    prob = prob/min(Hp,Hm) # from Pablo A Estevez et. al. Normalized Mutual Information Feature Selection

    # we still have to ensure that probabilities sum to 1
    prob = prob/sum(prob)
    return (prob, states)


def ENT_Probability(sess, net, states, input_sample,  args):

    ''' pass it forward and get network probabilities '''
    prob, states = net.forward_model(sess, states, input_sample)

    prob *= args.freqs
    prob = prob/sum(prob)

    return (prob,states)

def forward_with_mask(sess, net, states, input_sample, args):
    # global args
    # forward_args is a dictionary containing arguments for generating probabilities.
    # relevance = forward_args['relevance']
    # mask_reset_token = forward_args['mask_reset_token']
    # forbidden_token = forward_args['forbidden_token']
    # temperature = forward_args['temperature']
    # topn = forward_args['topn']

    # No relevance masking.
    if args.loss_model == 'NET': 
        # if args.relevance > 0.: 
        #     eprint( "NOTICE: cannot use relevance setting with NET loss!" ) 
        #     assert False
        prob, states = NET_Probability(sess, net, states, input_sample,args)
    elif args.loss_model == 'MMI': 
        prob, states = MMI_Probability(sess, net, states, input_sample, args)
    elif args.loss_model == 'ENT': 
        prob, states = ENT_Probability(sess, net, states, input_sample, args)
    elif args.loss_model == 'NORM': 
        prob, states = Norm_MMI_Probability(sess, net, states, input_sample, args)

    # Apply temperature.
    prob = scale_prediction(prob, args.temperature)


    # Apply top-n filtering if enabled

    # all but the topn highest probabilities are converted to 0
    # then the array is normalized 
    if args.topn > 0:
        prob[np.argsort(prob)[:-args.topn]] = 0 
        prob = prob / sum(prob)

    # freq = args.freqs
    # print (prob[7], prob[ord('#')],prob[32],prob[ord('e')])
    # print (freq[7], freq[ord('#')],freq[32],freq[ord('e')])
    # print (prob[7], prob[ord('#')],prob[32],prob[ord('M')])
    # print (freq[7], freq[ord('#')],freq[32],freq[ord('M')])
    # exit()

    # writer.close()
    return prob, states

# def beam2entropy(beam_outputs):

def beam_search_generator(sess, net, initial_state, initial_sample, early_term_token, beam_width, args):
    # global args

    '''Run beam search! Yield consensus tokens sequentially, as a generator;
    return when reaching early_term_token (newline).

    Args:
        sess: tensorflow session reference
        net: tensorflow net graph (must be compatible with the forward_net function)
        initial_state: initial hidden state of the net
        initial_sample: single token (excluding any seed/priming material)
            to start the generation
        early_term_token: stop when the beam reaches consensus on this token
            (but do not return this token).
        beam_width: how many beams to track
        forward_model_fn: function to forward the model, must be of the form:
            probability_output, beam_state =
                    forward_model_fn(sess, net, beam_state, beam_sample, forward_args)
            (Note: probability_output has to be a valid probability distribution!)
        tot_steps: how many tokens to generate before stopping,
            unless already stopped via early_term_token.
    Returns: a generator to yield a sequence of beam-sampled tokens.'''
    # Store state, outputs and probabilities for up to args.beam_width beams.
    # Initialize with just the one starting entry; it will branch to fill the beam
    # in the first step.
    beam_states = [initial_state] # Stores the best activation states
    beam_outputs = [[initial_sample]] # Stores the best generated output sequences so far.
    beam_probs = [1.] # Stores the cumulative normalized probabilities of the beams so far.
    beam_entps = [1.]

    count = 0
    while True:
        # Keep a running list of the best beam branches for next step.
        # Don't actually copy any big data structures yet, just keep references
        # to existing beam state entries, and then clone them as necessary
        # at the end of the generation step.
        new_beam_indices = []
        new_beam_probs = []
        new_beam_samples = []

        # Iterate through the beam entries.
        for beam_index, beam_state in enumerate(beam_states):

            beam_prob = beam_probs[beam_index]
            # beam_entp = beam_entps[beam_index]

            beam_sample = beam_outputs[beam_index][-1]

            # import pdb
            # pdb.set_trace()
            # Forward the model.
            prediction, beam_states[beam_index] = forward_with_mask( sess, net, beam_state, beam_sample, args) 

            # print (chr(beam_sample),end='')


            # print(sum(1 if p > 0. else 0 for p in prediction))
             # Sample best_tokens from the probability distribution.
            # Sample from the scaled probability distribution beam_width choices
            # (but not more than the number of positive probabilities in scaled_prediction).
            # count = min(beam_width, sum(1 if p > 0. else 0 for p in prediction))

            # best_probs = [p for p in np.random.choice(prediction,beam_width) if p > 0.]
            indexes = np.random.choice(len(prediction),size=args.beam_width,replace=False,p=prediction)
            best_iprobs = [(i,prediction[i]) for i in indexes if prediction[i] > 0.]

            # print(best_iprobs)
            # exit()
            # best_tokens = np.random.choice(len(prediction), size=args.beam_width, replace=False, p=prediction)
            for (token,prob) in best_iprobs:
            # for token in best_tokens:
                # prob = prediction[token] * beam_prob
                prob *= beam_prob
                if len(new_beam_indices) < beam_width:
                    # print('high prob***********')
                    # If we don't have enough new_beam_indices, we automatically qualify.
                    new_beam_indices.append(beam_index)
                    new_beam_probs.append(prob)
                    new_beam_samples.append(token)
                else:
                    # TAC -- simplified this code
                    # print('************low prob***********')
                    # Sample a low-probability beam to possibly replace.
                    
                    # np_new_beamhe_probs = np.array(new_beam_probs)
                    
                    # inverse_probs = -np_new_beam_probs + max(np_new_beam_probs) + min(np_new_beam_probs)
                    # inverse_probs = inverse_probs / sum(inverse_probs)
                    # sampled_beam_index = np.random.choice(beam_width, p=inverse_probs)
                    sampled_beam_index = np.argmin(new_beam_probs)
                    # print (len(sampled_beam_index))
                    # exit()
                    
                    if new_beam_probs[sampled_beam_index] <= prob:
                        # Replace it.
                        new_beam_indices[sampled_beam_index] = beam_index
                        new_beam_probs[sampled_beam_index] = prob
                        new_beam_samples[sampled_beam_index] = token

        # Replace the old states with the new states, first by referencing and then by copying.
        already_referenced = [False] * beam_width
        new_beam_states = []
        new_beam_outputs = []
        for i, new_index in enumerate(new_beam_indices):
            if already_referenced[new_index]:
                new_beam = copy.deepcopy(beam_states[new_index])
            else:
                new_beam = beam_states[new_index]
                already_referenced[new_index] = True
            new_beam_states.append(new_beam)
            new_beam_outputs.append(beam_outputs[new_index] + [new_beam_samples[i]])

        # Normalize the beam probabilities so they don't drop to zero
        beam_probs = new_beam_probs / sum(new_beam_probs)
        beam_states = new_beam_states
        beam_outputs = new_beam_outputs

        # Prune the agreed portions of the outputs
        # and yield the tokens on which the beam has reached consensus.
        l, early_term = consensus_length(beam_outputs, early_term_token)

        # sanity check against models (line frequency alone) that never find
        # termination characters  
        # count += 1
        # if ( count >= 30 ): early_term = True

        #################################################
        # just display the probs as strings 
        # import string
        if args.verbose:
            trans = str.maketrans("\n", u"\u00B7") # center dot for display only
            print ('\n','-'*30)
            strings = []
            for i,bo in enumerate(beam_outputs):
                s = ''.join([chr(i) for i in bo])
                p = beam_probs[i]
                e = sentenceEntropy(s,args)
                strings.append('{:.4f} - {}- {}'.format(p,s,e))
            for s in sorted(strings):
                print(s.translate(trans))
            if l>0:
                print('{:6s} - {}'.format(' ','='*l))
            print('.'*30)
        #################################################

        if l > 0:
            for token in beam_outputs[0][:l]: 
                yield token
            beam_outputs = [output[l:] for output in beam_outputs]
        if early_term: return

def sentenceComplexity(sentence):
    return sentence.count(' ')/len(sentence)

def sentenceEntropy(sentence,args):
    return sum([args.freqs[ord(i)] for i in sentence])


if __name__ == '__main__':
    main()
