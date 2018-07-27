from difflib import SequenceMatcher

class longest:
    def __init__():
def longestSubstring(strings):

    result = ''

    for i in range(len(strings[0])):

    # initialize SequenceMatcher object with 
    # input string
    seqMatch = SequenceMatcher(None,str1,str2)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))
 
    return str1[match.a: match.a + match.size]
 
# Driver program
if __name__ == "__main__":
    str1 = [.23,.45,.23,.123,.456]
    str2 = [.23,.45,.23,.423,.556]

    print(    longestSubstring(str1,str2))
