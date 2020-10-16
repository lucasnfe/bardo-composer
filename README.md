# Computer-Generated Music for Tabletop Role-Playing Games

This repository contains the source code to reproduce the results of the [AIIDE'20](https://webdocs.cs.ualberta.ca/~santanad/aiide/)
paper [Computer-Generated Music for Tabletop Role-Playing Games](https://arxiv.org/abs/2008.07009).
This paper presents *Bardo Composer*, a system to generate background music for tabletop role-playing games. Bardo Composer uses a
speech recognition system to translate player speech into text, which is classified according to a model of emotion. Bardo Composer
then uses Stochastic Bi-Objective Beam Search, a variant of Stochastic Beam Search that we introduce in this paper, with a neural 
model to generate musical pieces conveying the desired emotion.

## Examples of Generated Pieces

- [Piece 1](https://raw.githubusercontent.com/lucasnfe/bardo-composer/master/output/piece1_ag_sus.wav)
- [Piece 2](https://raw.githubusercontent.com/lucasnfe/bardo-composer/master/output/piece2_ag_calm.wav)
- [Piece 3](https://raw.githubusercontent.com/lucasnfe/bardo-composer/master/output/piece3_sus_ag.wav)

## Installing Dependencies

## Reproducing Results

Bardo Composer uses a fine-tuned BERT to classify the story emotion and a fine-tuned GPT2 to classify the music emotion.
In the paper, we report the accuracy of these models. This section describes how to reproduce the results we found.

### Story Emotion Classification

We compared the fine-tuned BERT model for story emotion classification with the simpler Naıve Bayes approach of [Padovani, Ferreira, and
Lelis (2017)]. 

#### Download Pre-trained BERT

#### Fine-tune Pre-trained BERT

### Music Emotion Classification

We compared the fine-tuned GPT2 model for music emotion classification with the simpler LSTM approach of [Ferreira and Whitehead (2019)].

#### Download Pre-trained Models

- LSTM
- GPT2

#### Fine-tune Pre-trained Models
