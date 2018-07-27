from __future__ import print_function
 
import numpy as np
import tensorflow as tf

import argparse
import os
import pickle
import copy
import sys
import html

from utils import TextLoader
from model import Model

from chatbot import *

from utils import eprint # print to stderr
from utils import get_paths, get_args

from utils import args

from utils import Samples
samples = Samples()

def main():
    global args
    args = get_args()

    print("Creating model...")
    args.saved_args.batch_size = args.beam_width
    net = Model(args.saved_args, True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Make tensorflow less verbose; filter out info (1+) 
    # and warnings (2+) but not errors (3).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    with tf.Session(config=config) as sess:

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(net.save_variables_list())

        # Restore the saved variables, replacing the initialized values.
        print("Restoring weights...")
        saver.restore(sess, args.model_path)
        
        max_length = args.n
        beam_width = args.beam_width
        relevance = args.relevance
        temperature = args.temperature
        topn = args.topn

        chat1_states = initial_state_with_relevance_masking(net, sess, relevance)
        chat2_states = initial_state_with_relevance_masking(net, sess, relevance)

        response = args.initial_text # anything to start a dialog
        for i in range(args.count):

            # which test to do
            if args.test_mode == 'QA':
                response = samples.get()
                print ('Q:',response)
                chat1_states, response = get_response(response, chat1_states, net, sess, args)
                print ('A:',response)

            if args.test_mode == '1CHAT': # two chat bots talking 
                print ('S1:',response)
                chat1_states, response = get_response(response, chat1_states, net, sess, args)
                print ('A1:',response)

            if args.test_mode == '2CHAT': # two chat bots talking 
                chat1_states, response = get_response(response, chat1_states, net, sess, args)
                print ('12:',response)
                chat1_states, response = get_response(response, chat2_states, net, sess, args)
                print ('22:',response)

def get_response(user_input, states, net, sess, args):
    
    # if reset: states = initial_state_with_relevance_masking(net, sess, args.relevance)

    states = forward_text(net, sess, states, args.relevance, sanitize_text("> " + user_input + "\n>"))

    computer_response_generator = beam_search_generator(sess=sess, net=net,
        initial_state=copy.deepcopy(states), initial_sample=ord(' '),
        early_term_token=ord('\n'), beam_width=args.beam_width, args=args)

    parts = []
    out_chars = []
    # print('\n>')

    for i, char_token in enumerate(computer_response_generator):
        out_chars.append(chr(char_token))
        parts.append(possibly_escaped_char(out_chars))
        if args.verbose:
            print(possibly_escaped_char(out_chars), end='', flush=True)
        states = forward_text(net, sess, states, args.relevance, chr(char_token))
        if i >= args.max_length: break
    states = forward_text(net, sess, states, args.relevance, sanitize_text("\n> "))
    # user_input = ''.join(parts)
    return (states, ''.join(parts[1:])) # [1:] ignores initial_sample 
    # print('\n>{}'.format(user_input))


if __name__ == '__main__':
    main()