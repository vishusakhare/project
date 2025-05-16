def string_both_ends(str):
    print('vaishnavi')
    if len(str) < 2:
        return ''
    return str[0:2] + str[-2:]
print(string_both_ends('vaishnavi'))  # Output: 'w3ce'
print(string_both_ends('vaish'))          # Output: 'w3w3'
print(string_both_ends('v'))




inputString = "vaishnavi"
count = 0
 

for i in inputString:
      count = count + 1
newString = inputString[ 0:2 ] + inputString [count - 2: count ] 
 

print("Input string = " + inputString)
print("New String = "+ newString)



str="sharda vavi"
print("Given String :",str)
ch = str[0]
str = str.replace(ch, '$')
str = ch + str[1:]
print("After String :",str)


#Write a decorator function to do the following items on your base_model of the car:
Add a functionality to have alloy wheel to client’s existing car.
Add a functionality to have sunroof to client’s existing car.
Add a functionality to have alloy wheels + sunroof to client’s existing car.

def enhance_car(func):
    def wrapper(car):
        print(f"Enhancing {car['base_model']} with:")
        func(car)
        print("Enhancements completed.\n")
    
    return wrapper

#@enhance_car
def add_alloy_wheels(car):
    car['alloy_wheels'] = True
    print("- Alloy Wheels")

#@enhance_car
def add_sunroof(car):
    car['sunroof'] = True
    print("- Sunroof")

#@enhance_car
def add_alloy_wheels_and_sunroof(car):
    car['alloy_wheels'] = True
    car['sunroof'] = True
    print("- Alloy Wheels + Sunroof")

# Example usage:
car = {'base_model': 'Toyota Camry'}

add_alloy_wheels(car)
add_sunroof(car)
add_alloy_wheels_and_sunroof(car)

print("Final car details:")
print(car)
