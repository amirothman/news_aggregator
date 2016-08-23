#!/bin/bash

echo $1 | ./fasttext print-vectors model/fast_text.bin > textfiles/fast_text_word_vectors.txt
