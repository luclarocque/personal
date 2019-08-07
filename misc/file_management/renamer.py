import os, re

def main():
    os.chdir(r"C:\Users\Luc\Google Drive\MyBoy")
    for f in os.listdir():
        pat = re.compile(".* - (.* - .*) \(U\)\(TrashMan\).(.*)")
        m = pat.search(f)
        if m:
            name = m.group(1)
            ext = m.group(2)
            newname = name + "." + ext
            os.rename(f, newname)


if __name__ == "__main__":
    main()