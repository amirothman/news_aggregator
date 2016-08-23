#!/bin/bash

./fasttext print-vectors model/fast_text.bin < textfiles/temp_fast_text > textfiles/fasttext_word_vector.txt
