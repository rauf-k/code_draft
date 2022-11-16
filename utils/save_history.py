
def save_history(hist):
    list_of_chars = ['[', ']', '{', '}']

    for ch in list_of_chars:
        hist = hist.replace(ch, '')

    file1 = open("res/training_metrics.log", "a")
    file1.write(hist + "\n")
    file1.close()
