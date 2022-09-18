import asyncio
from signal import SIGINT, SIGTERM
from sys import executable as python_exe

async def gentle_timeout(proc, interrupt, terminate=1.0, kill=0.5, input=None):
    comm = proc.communicate(input)
    try:
        result = await asyncio.wait_for(
            proc.communicate(input),
            timeout=interrupt)
    except asyncio.TimeoutError:
        print(f"Timed out after {interrupt} seconds: sending interrupt")
        proc.send_signal(SIGINT)
    else:
        return result

    try:
        result = await asyncio.wait_for(proc.communicate(),
                    timeout=terminate)
    except asyncio.TimeoutError:
        print(f"Timed out *AGAIN* after {terminate} seconds")
        proc.send_signal(SIGTERM)
    else:
        return result

    try:
        result = await asyncio.wait_for(proc.communicate(),
                    timeout=kill)
    except asyncio.TimeoutError:
        print(f"Set phasers to kill after {kill} seconds")
        proc.send_signal(SIGKILL)
    else:
        return result

    print("Awaiting communication")
    result = await proc.communicate()
    return result

async def run(i, sleep_sec):
    print(i, "Creating process")
    proc = await asyncio.create_subprocess_exec(
        python_exe, "sleep.py", str(sleep_sec),
        stdout=asyncio.subprocess.PIPE
    )
    print("Communicating", sleep_sec)
    result = await gentle_timeout(proc, interrupt=2, terminate=2)
    print(i, "return code:", proc.returncode)
    print(i, "result:", result)


async def main():
    result = await asyncio.gather(*(run(*args)
        for args in enumerate([0.5, 1, 3, 5])))

asyncio.run(main())
