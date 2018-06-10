# number-tracking
ConvNet that recognizes and tracks hadwritten digits via webcam

CoLab: https://colab.research.google.com/drive/1ZCfa4W4Z1adiNfTEwMmmxbZWlIKVafZh

TODO main loop:
1. Look for paper
2. Extract digits
3. Track and aggregate confidences for first N frames
4. Just track; when a digit is lost restart main loop
