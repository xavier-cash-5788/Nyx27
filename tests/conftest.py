import os
import sys

# Permet `import nyx` depuis les tests, où que pytest soit lancé.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
