import os, sys, getopt


def id_to_emo(emo_id):

  d = {'0':'neutral',
       '1':'anger',
       '2':'disgust',
       '3':'fear',
       '4':'happiness',
       '5':'sadness',
       '6':'surprise',
        }

  return d[emo_id]

def join_with_alternate_prefixes(dials, emo_ids):
    result = []
    prefixes = ["#Farnoosh", "#Hasan"]
    prefix_index = 0

    for i, dial in enumerate(dials):
        emotion = id_to_emo(emo_ids[i])
        prefix = prefixes[prefix_index % 2]
        result.append(prefix + ' $' + emotion + ': ' + dial)
        prefix_index += 1

    return ''.join(result)



def parse_data(in_dir):

    mode = ''

    # Finding files
    if in_dir.endswith('train'):
        mode = 'train'
        dial_dir = os.path.join(in_dir, 'dialogues_train.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_train.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_train.txt')
    elif in_dir.endswith('validation'):
        mode = 'test'
        dial_dir = os.path.join(in_dir, 'dialogues_validation.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_validation.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_validation.txt')
    elif in_dir.endswith('test'):
        mode = 'test'
        dial_dir = os.path.join(in_dir, 'dialogues_test.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_test.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_test.txt')
    else:
        print("Cannot find directory")
        sys.exit()


    # Open files
    in_dial = open(dial_dir, 'r')
    in_emo = open(emo_dir, 'r')
    in_act = open(act_dir, 'r')

    # Create Dictionary
    d = {"text":[], "emotions":[], "acts":[], "text_label":[]}


    for line_count, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):
        
        emos = line_emo.split(' ')
        emos = emos[:-1]
        d['emotions'].append(emos)

        seqs = line_dial.split('__eou__')
        seqs = seqs[:-1]
        if mode=='test':
          label = seqs.pop()        
          seqs.append(" ")
        else:
          label = None

        d['text_label'].append(label)

        seqs = join_with_alternate_prefixes(seqs, emos)
        d['text'].append(seqs)

        acts = line_act.split(' ')
        acts = acts[:-1]
        d['acts'].append(acts)
        

    return d


def main(argv):

    in_dir = ''

    try:
        opts, args = getopt.getopt(argv,"h:i:",["in_dir="])
    except getopt.GetoptError:
        print("python3 parser.py -i <in_dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("python3 parser.py -i <in_dir>")
            sys.exit()
        elif opt in ("-i", "--in_dir"):
            in_dir = arg


    print("Input directory : ", in_dir)

    data = parse_data(in_dir)
    print("len data", len(data['text']), len(data['acts']))
    print("sample 10", data['text'][1])
    print("label 10", data['text_label'][1])


if __name__ == '__main__':
    main(sys.argv[1:])
