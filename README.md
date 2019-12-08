# Text-Generation-Network

This was a personal introductory mini project for Deep LearningI did 
for Rensselaer Center for Open Source, where I coded a Long Short Term Memory 
to intake the texts from one of my favorite books, Huckleberry Finn. Since Huckleberry Finn
has a decently large size of characters, I had enough sample to develop
a relatively accurate network, with slight mispellings periodically. 

I also ran the Dark Knight script to see how my network would perform on a
rather spacious text. However, the output wasn't satisfactory.

The trained my 3 layer network over 25 Epochs. After that, the model started
to overfit and produce weird results such as reptition of characters. It uses
simple character to integer mapping, and standard loss back propagation to determine
expected outputs and further network updates.

Huge Credits to:
Jason Brownlee for the idea and tutorial
Sentdex, amazing youtuber with amazing tutorials
My RCOS group, Machine Ex Machina for supervising and assisting me
