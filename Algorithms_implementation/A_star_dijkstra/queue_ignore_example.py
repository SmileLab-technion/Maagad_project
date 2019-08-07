import heapq

# this is an example file to demonstrate how to ignore items when popping from heapq.
# the queue is a collection of pairs, the first item is the priority (integer), and the second item is the element (in
# this example strings).

# start with an empty queue
q = []

# assume that we insert an element "a" to the queue with priority 100.
heapq.heappush(q, (100, "a"))

# afterwards we insert a new element "b" with priority 99.
heapq.heappush(q, (99, "b"))

# now we want to "update" the first item to priority 98
heapq.heappush(q, (98, "a"))

# what we want is to have two items in the following order (98, "a") and (99, "b") but instead we have three items:
# (98, "a"), (99, "b") and (100, "a"). however, we know that after popping "a" once, the next time we pop an "a" we can
# ignore it. lets save an ignore set
ignore = set()

# this is how we ignore items:
while len(q) > 0:
    current_priority, current_item = heapq.heappop(q)
    if current_item in ignore:
        print("ignored {} with priority {}".format(current_item, current_priority))
    else:
        ignore.add(current_item)
        print("did not ignore {} with priority {}".format(current_item, current_priority))
