import threading
import time

def walk_dog(first):
    time.sleep(8)
    print(f"You finish walking {first}")

def take_out_trash():
    time.sleep(2)
    print("You take out the trash")

def get_mail(first, last):
    time.sleep(4)
    print(f"You get the mail from {first} {last}")

chore1 = threading.Thread(target = walk_dog, args=("Scooby",))
chore1.start()

chore2 = threading.Thread(target = take_out_trash)
chore2.start()

chore3 = threading.Thread(target = get_mail, args=("Vayishu", "Pandey"))
chore3.start()

# If we don't join the threads, the main program may finish before the threads complete
chore1.join()
chore2.join()
chore3.join()

print("All chores are completed")

"""
Output:
You take out the trash
You get the mail
You finish walking the Dog
All chores are completed
"""