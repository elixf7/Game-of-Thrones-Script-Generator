# Game of Thrones Script Generator
![IMG](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMTRkaGdrbWsydzZqemlsMmY4c3E1N2RqNzZkeWZ0aDB5dG0xemRoeiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3oEjI1erPMTMBFmNHi/giphy.gif)

This README is an outline of the steps taken to create this project. More detailed comments are provided in the code for specifics. This project has a single ipynb file for the code called **ai-GOT-script-generator.ipynb**. This file handles all of the code from processing the txt files to producing an output that imitates the game of thrones script. The **Data** folder contains all of the txt files for GOT episodes downloaded from shekharkoirala on github [HERE](https://github.com/shekharkoirala/Game_of_Thrones). The **Training** file is a directory that is meant to save the weights of a model after each training epoch. There are some preloaded weights in this directory if the user wishes to start with those. Finally, the **merged_file.txt** contains all of the scripts combined into a single txt file. 

### Overview
1. Introduction
2. Processing the Text
3. Analyzing the Script
4. Building the Model
5. Results
6. Future Additions


### Introduction
Game of Thrones is one of my favorite shows, and after looking at other text generators I thought this would be the perfect TV show to try and create a script for. The show has eight seasons, and plenty of diverse characters and vocabulary to make for an interesting project. The recurrent neural network in this notebook is not particularly large or complex to avoid significantly long training times, but the output is still understandable and quite entertaining. 

In addition to training the model on all 8 seasons of script, I added a feature that allows the model to be trained on dialog from any character in the show. With enough training, this should produce a model that produce text similar to the dialog of **Jon Snow** for example. 

More information about the series can be found on the [Game of Thrones Wikipedia](https://en.wikipedia.org/wiki/Game_of_Thrones)

### Processing the Text
The first step was to combine all of the individual episode txt files into a single txt file that could be read. The following code segment acheives this and stores the result in **merged_file.txt** for easier acess.

```
inputs = []
for file in os.listdir("Data"):
    if file.endswith(".txt"):
        inputs.append(os.path.join("Data", file))

with open('merged_file.txt', 'w') as outfile:
    for fname in inputs:
        with open(fname, encoding="utf-8", errors='ignore') as infile:
            outfile.write(infile.read())
```
The next step was to remove any unnecary characters such as the carriage return "\r" and tabs"\t" and to strip the whitespace from the ends of each line in the file. This way the model will not be trained to produce excess whitespace. 

### Analyzing the Script
The enitre GOT script is 2,474,457 characters long with 95 unique characters after removing the symbols descirbed above. this is a very robust amount of data that will make training the model easier and less likely to overfit. 

