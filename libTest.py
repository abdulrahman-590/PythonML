# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


# # plt.figure(figsize=(2,2), dpi=300)

# x = [0,1, 2,3]
# y = [0,1,2,3]



# # plt.plot(x, y , color='red', linestyle=':', linewidth=2,
# #          marker='o', markersize = 5)


# plt.plot(x, y, "ro--") #shorthand notation

# plt.title("Graph 01")
# plt.xlabel("X-Axis")
# plt.ylabel("Y-Axis")

# # plt.xticks([0,1,2,3,4])
# # plt.yticks([0,1,2,3,4])

# plt.xticks(range(0,11, 10))


# plt.legend() # keymap (add label in plot() for this to work)
# plt.show()


# # Bar Chartß

# # plt.figure(figsize=(2,2), dpi=300)

# labels = ["A", "B", "C"]
# values = [1,4,2]
# bars = plt.bar(labels, values)
# bars[0].set_hatch('/')
# bars[1].set_hatch('o')
# bars[2].set_hatch('*')



# # plt.savefig("graph.png", dpi=300)

# plt.show()

test = [0,1,2,3,4,5,6,7]
test2 = [0,1,2,3,4,5,6,7]
res = test.extend(test2)
print(test)