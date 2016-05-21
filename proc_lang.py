
def process_chunks(rl, part):
    import nltk
    tmp_itemlist = set()
    tmp_desclist = set()

## For the description text, we should do some manipulation beforehand, like removing bad characters ( '/' for example, and '*')

    for i in rl['menu']:
        #tkns = nltk.word_tokenize(i['item'].lower())
        tkns = nltk.word_tokenize(i[part].lower())

        if tkns:
            try:
                tagged = nltk.pos_tag(tkns)
                chunkGram = r"""Chunk: {<N.+>*}""" # This will find any number of nouns in a row, including 1
                chunkParser = nltk.RegexpParser(chunkGram)
                chunked = chunkParser.parse(tagged)
            except Exception as e:
                print(str(e))

            wrds = []
            for subtree in chunked.subtrees():
                wrds.append(" ".join([i[0] for i in subtree.leaves()]))

            for i in range(1,len(wrds)): # Append the word to tmp_itemlist only if it is unique
                tmp_itemlist.add(wrds[i])

    return(tmp_itemlist)


def create_factors():
    ##Create the factors for each restaurnt here, and compile them all in the driver file
    pass
