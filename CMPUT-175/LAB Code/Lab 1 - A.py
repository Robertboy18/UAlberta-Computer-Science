# 1
# Create the dictionary for the prices
spring_Flower_Bulb = dict(
    [("daffodil", 0.35), ("tulip", 0.33), ("crocus", 0.25), ("hyacinth", 0.75), ("bluebell", 0.50)])

# 2
# Create the second dictionary for the number of items
bluebell_Greenhouses = dict([("daffodil", 50), ("tulip", 100)])

# 3
# increase the price and round it up to 2 decimal places
spring_Flower_Bulb["tulip"] = round((spring_Flower_Bulb["tulip"] * 1.25), 2)

# 4
# update the second dictionary with the new value
bluebell_Greenhouses.update([("hyacinth", 30)])

# 5
"""print("{:<5s}".format("bulb code") + " * " + "{:4s}".format("number of bulbs") + " = " + "$" + "{:6s}".format(
"subtotal"))"""
# bluebell_Greenhouses.update([("bluebell", 0), ("crocus", 0)])
spring_Flower_Bulb_1 = dict(sorted(spring_Flower_Bulb.items(), key=lambda x: x[0].lower()))  # sort the first one
bluebell_Greenhouses_1 = dict(sorted(bluebell_Greenhouses.items(), key=lambda x: x[0].lower()))  # sort the second one
"""
for (key1, value1), (key2, value2) in zip(spring_Flower_Bulb_1.items(), bluebell_Greenhouses_1.items()): product = 
value1 * value2  # find the value of the product if product != 0:  # if it aint equal to 0 print print("{
:<5s}".format(key2[0:3].upper()) + " " * 12 + "{:4s}" .format(str(value2)) + " " * 10 + "$  " + str(value1)) 
"""

# 6
# Print the desired output
print("You have purchased the following bulbs: ")
total = 0  # initiate the variable to 0
for (key1, value1), (key2, value2) in zip(spring_Flower_Bulb_1.items(), bluebell_Greenhouses_1.items()):
    product = value1 * value2  # find the value of the product
    total += product
    if product != 0:  # if it aint equal to 0 print
        print("%-5s" % (key2[0:3].upper()) + " * " + "%4s" % str(value2) + " = $ " + "%1.2f" % product)
print("Thank you for purchasing 180 bulbs from Bluebell Greenhouses. ")
print("Your total comes to $ " + str(total) + ".")
