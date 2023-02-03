import importlib
import sys
from pathlib import Path

def main():
  """use a main function so that the real main module has a fully qualified
  name (thus will not be imported again)"""
  if len(sys.argv) == 1:
      print(f'usage: {sys.argv[0]} <submodule> <submodule args ...>')
      return 2
  name = sys.argv[1]
  del sys.argv[1]
  sys.argv[0] = name
  mod = Path(__file__).resolve().parent.name
  return importlib.import_module(f'{mod}.{name}').main()

if __name__ == '__main__':
  sys.exit(main())
