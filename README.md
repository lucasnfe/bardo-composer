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

#### Fine-tune Pre-trained BERT

```
cd composer/clf_dnd/
python3 clf_bert.py --data ../../data/dnd/
```

#### Train Naive Bayes 

```
cd composer/clf_dnd/
python3 clf_nbayes.py --data ../../data/dnd/
```

Both these scripts will perform and report accuracy experiments on the Call of the Wild dataset [Padovani, Ferreira, and
Lelis (2017)]. 

### Music Emotion Classification

We compared the fine-tuned GPT2 model for music emotion classification with the simpler LSTM approach of [Ferreira and Whitehead (2019)].

#### Download Pre-trained GPT-2

In the paper, the GPT-2 model was pre-trained using a new dataset called [ADL-Piano-Midi](). The pre-trained model can be download as follows:

```
$ wget https://github.com/lucasnfe/bardo-composer/releases/download/0.1/pre-trained-gpt2.zip
```

#### Fine-tune Pre-trained GPT-2

```
cd composer/clf_vgmidi/
python3 clf_gpt2.py --conf clf_gpt2.json
```

#### Train LSTM 

```
cd composer/clf_vgmidi/
python3 clf_lstm.py --conf clf_lstm.json
```
