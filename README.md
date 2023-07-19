# Game of Thrones Script Generator
![IMG](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMTRkaGdrbWsydzZqemlsMmY4c3E1N2RqNzZkeWZ0aDB5dG0xemRoeiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3oEjI1erPMTMBFmNHi/giphy.gif)

This README is an outline of the steps taken to create this project. More detailed comments are provided in the code for specifics. This project has a single ipynb file for the code called **ai-GOT-script-generator.ipynb**. This file handles all of the code from processing the txt files to producing an output that imitates the game of thrones script. The **Data** folder contains all of the txt files for GOT episodes downloaded from shekharkoirala on github [HERE](https://github.com/shekharkoirala/Game_of_Thrones). The **Training** file is a directory that is meant to save the weights of a model after each training epoch. There are some preloaded weights in this directory if the user wishes to start with those. Finally, the **merged_file.txt** contains all of the scripts combined into a single txt file. 

### Overview
1. Introduction
2. Loading the Text
3. Analyzing the Script
4. Processing the Characters
5. Building the Model
6. Results
7. Future Additions


### Introduction
Game of Thrones is one of my favorite shows, and after looking at other text generators I thought this would be the perfect TV show to try and create a script for. The show has eight seasons, and plenty of diverse characters and vocabulary to make for an interesting project. The recurrent neural network in this notebook is not particularly large or complex to avoid significantly long training times, but the output is still understandable and quite entertaining. 

In addition to training the model on all 8 seasons of script, I added a feature that allows the model to be trained on dialog from any character in the show. With enough training, this should produce a model that produce text similar to the dialog of **Jon Snow** for example. 

More information about the series can be found on the [Game of Thrones Wikipedia](https://en.wikipedia.org/wiki/Game_of_Thrones)

### Loading the Text
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

I then separated the script into lines by specific characters. To do this, I had to solve several challenges:
1. How to tell if a line is dialog or just setting a scene
2. How to differentiate a character name and a word
3. Count the number of lines per character, and store these lines elsewhere

This was done by recognizing all character dialog starts with the character name in all caps followed by a semicolon. For example: **CERCEI: I love my children**
I was then able to iterate over each line storing the characters in a `name` variable. I stopped storing characters once I encountered a semicolon, and cleared the name variable if I reached the end of the line. Furthermore, to fix encountering a semicolon in normal dialog, I cleared the name variable once the string was 20 characters long (since nobody has a name that long). I then stored the line in a dictionary with the key being the name variable, and the value being a concatenated string of all the character lines. Even after some more tranformations (detailed in the code), the final dictionary still had some keys that were not people in the show. One example is **EXT** which means "exit" in the script. These few mistakes were removed manually.

The final dictionary was converted to a list and sorted in descending order. Characters below a specified threshold of lines were dropped. The results were plotted as seen in the graph below. To my suprise Tyrion had the most lines in the series by far!

### Processing the Characters
A simple but important step in the process is to assign a numerical index to each unique character in the text. Since the script had 95 unique characters, each of these was given a number in the range 0-94. For instance the letter 'a' is represented by a 56 in this case. Then the entire text can be converted to an array of numbers instead of a long string. This is important for training the model.

Also, a sequence length of 100 was chosen. This simply means that the model will have 100 characters of previous context when it is trying to predict the next letter. The text therefore has to be split up into chunks of 100 characters each which I did. 
```
char_to_idx = dict((c, i) for i, c in enumerate(vocab))
idx_to_char = np.array(vocab)
text_as_int = np.array([char_to_idx[c] for c in text])
```

Next I created the `input_text` and `target_text` variables which are misaligned by one character. This is necessary so that the model can be trained to predict the next character in the sequence. For instance, if the input is **Hello World**, then the output would be **ello World!**. As you can see, at index 0 the input is **H** and the output is **e**. So, the model is being trained to ouput an e after the H. 

### Building the Model


