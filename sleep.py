import sys
import time

seconds = float(sys.argv[1])
print(f"Sleeping for {seconds} seconds")

try:
    time.sleep(seconds)
except KeyboardInterrupt:
    seconds /= 2
    print(f"Interrupted: sleeping {seconds} more second")
    time.sleep(seconds)
    print("Snoozed!")
    sys.exit(1)
else:
    print("Slept!")
