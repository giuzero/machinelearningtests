def blackjack_hand_greater_than(hand_1, hand_2):
    """
    Return True if hand_1 beats hand_2, and False otherwise.
    
    In order for hand_1 to beat hand_2 the following must be true:
    - The total of hand_1 must not exceed 21
    - The total of hand_1 must exceed the total of hand_2 OR hand_2's total must exceed 21
    
    Hands are represented as a list of cards. Each card is represented by a string.
    
    When adding up a hand's total, cards with numbers count for that many points. Face
    cards ('J', 'Q', and 'K') are worth 10 points. 'A' can count for 1 or 11.
    
    When determining a hand's total, you should try to count aces in the way that 
    maximizes the hand's total without going over 21. e.g. the total of ['A', 'A', '9'] is 21,
    the total of ['A', 'A', '9', '3'] is 14.
    
    Examples:
    >>> blackjack_hand_greater_than(['K'], ['3', '4'])
    True
    >>> blackjack_hand_greater_than(['K'], ['10'])
    False
    >>> blackjack_hand_greater_than(['K', 'K', '2'], ['3'])
    False
    
    
    """
    print("\n-------------------------------------------------")
    print("{} - {}".format(hand_1, calc_hand_value(hand_1)))
    print("{} - {}".format(hand_2, calc_hand_value(hand_2)))
    print("#################################################")
    return (calc_hand_value(hand_1)>calc_hand_value(hand_2) or calc_hand_value(hand_2)>21)and calc_hand_value(hand_1)<=21
    #help(hand_1)
    #return hand_1.__gt__(hand_2)

def calc_hand_value(hand):
    hand_value = 0
    for card in hand:
        if card == 'J' or card == 'Q' or card == 'K':
            hand_value += 10
        if card.isdigit():
            hand_value += int(card)
    aces = sum([x=='A'for x in hand])
    #print("BEFORE ACES: {} - {} - {}".format(hand, hand_value, aces))
    hand_value += aces
    while hand_value + 10 <= 21 and aces > 0:
        hand_value += 10
        aces -= 1
    return hand_value

def chap1():
    spam_amount = 0
    print(spam_amount)

    # Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
    spam_amount = spam_amount + 4

    if spam_amount > 0:
        print("But I don't want ANY spam!")

    viking_song = "Spam " * spam_amount
    print(viking_song)

    print(5 // 2)
    print(6 // 2)

    print(type(spam_amount))

    print(min(1, 2, 3))
    print(max(1, 2, 3))

    print(abs(32))
    print(abs(-32))

    print(float(10))
    print(int(3.33))
    print(int('807') + 1)

def word_search(doc_list, keyword):
    """
    Takes a list of documents (each document is a string) and a keyword. 
    Returns list of the index values into the original list for all documents 
    containing the keyword.

    Example:
    doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
    >>> word_search(doc_list, 'casino')
    >>> [0]
    """
    findings = []
    for doc in doc_list:
        for word in doc.split(" "):
            if keyword.lower() in word.lower() and (len(word)==len(keyword) or (len(word)-1==len(keyword))) and doc_list.index(doc) not in findings:
                findings.append(doc_list.index(doc))
                
    return findings
            
def count_negatives(nums):
    return len([num for num in nums if num < 0]) #or return sum([num < 0 for num in nums])

def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    return [x>thresh for x in L]

def test_lambda(x):
    return lambda a : a * x

tripler = test_lambda(3)
print(tripler(8))
