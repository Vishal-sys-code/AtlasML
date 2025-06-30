# **Pass by value vs Pass by Reference**

Python passes a ref to the object, not the actual object itself, and not a copy. However, whether changes inside the function affect the original object depends on whether the object is mutable or immutable.

In Python, the term pass by value can be misleading, because python does not use traditional pass by value like in C or Java.

```cpp
void modify(int x){
    x = 10
}
int main(){
    int a = 5;
    modify(a);
    // a is still 5
}
```

What actually python do?
You are passing a reference to the object (not a copy). But you cannot rebind that reference to affect the original in the calling scope. If the object is immutable, any modification creates a new object, not affecting the original.

There are two types of objects in Python:
- **Mutable objects**: Lists, Dictionarires, Set, custom objects
- **Immutable Objects**: int, str, float, tuple

### **For Immutable Types:**
- Python passes a reference to the object.
- But since the object is immutable, you can’t modify it in place, so it behaves like pass-by-value.

```python
def modify(x):
    x = x + 10
    print("Inside function:", x)

a = 5
modify(a)
print("Outside function:", a)

"""
Output:
Inside function: 15
Outside function: 5
"""
```

### **For Mutable Types:**
Python passes a reference, and since the object is mutable, changes inside the function affect the original.

```py
def modify(lst):
    lst.append(10)
    print("Inside function:", lst)

my_list = [1, 2, 3]
modify(my_list)
print("Outside function:", my_list)

"""
Output:
Inside function: [1, 2, 3, 10]
Outside function: [1, 2, 3, 10]
"""
```

### **Summary:**
- Python does not do classic pass-by-value or pass-by-reference.
- It always passes a reference to the object, but what you can do with that reference depends on whether the object is mutable.  

<hr>

# **GIL [Global Interpreter Lock]**
It is a mutex(mutual exclusion lock) used in CPython to ensure that only one thread executes Python bytecode at a time, even on multi-core processors.

It was introduced to:
- Prevent Race Conditions in memory management.
- Simplify the implementation of python interpreter.
- Ensure safety and correctness without requiring devs to manage locking at manually.

```python
# Use multiprocessing for CPU-bound tasks instead of multithreading
from multiprocessing import Process

def compute():
    # CPU-intensive task
    pass

if __name__ == "__main__":
    p1 = Process(target=compute)
    p2 = Process(target=compute)
    p1.start()
    p2.start()
```

<hr>

# **MultiThreading**
MultiThreading = Used to perform multiple tasks concurrently(multitasking). Good for I/O bounds tasks like reading files or fetching data from APIs threading. Thread(target = my_function)

