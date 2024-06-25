import importlib
MCM = importlib.import_module("MC_Pro-main")
from MCM import MC_main as MC

#from mc_pro import MC_main as MC
#importlib.reload(MC)

print("If you can read this you're probably a developer.")
print("Did I guess right? Ha. I knew it.")
print("I bet your name is Svetlana. Am I right?")
print("Well, at least I got the first one right. Is it Gwendelen?")
bl_info = {
    "name": "MC_main",
    "author": "Rich Colburn, email: CharacterPhysics101@gmail.com",
    "version": (1, 0),
    "blender": (4, 1, 0),
    "location": "View3D > Extended Tools > Modeling Cloth",
    "description": "It's like cloth but in a computer!",
    "warning": "When you feel like you're being watched, you're not actually a cartoon.",
    "wiki_url": "",
    "category": '3D View',
}

   
def register():
    MC.register()

    
def unregister():
    MC.unregister()
    
#-----------------------------------------------
# |       Commercial Use License Agreement     |
#-----------------------------------------------

# This Commercial Use License Agreement ("Agreement") is entered into
#    by and between CharacterPhysics.com ("Licensor") and the individual
#    or entity agreeing to these terms ("Licensee").

# 1. Grant of License: Licensor hereby grants to Licensee a non-exclusive,
#    non-transferable license to use the Blender addon created by Licensor
#    ("Addon") for commercial purposes.

# 2. Permitted Use: Licensee may use the Addon to create,
#    modify, and distribute derivative works for commercial use.

# 3. Restrictions: Licensee shall not sublicense, sell, or distribute the
#    Addon or any part of it without prior written consent from Licensor.

# 4. Ownership: Licensor retains all right, title, and interest in and to the
#    Addon, including all intellectual property rights.

# 5. Warranty: The Addon is provided "as is," without warranty of any kind,
#    express or implied. Licensor disclaims all warranties, including but not
#    limited to the implied warranties of merchantability and fitness for a
#    particular purpose.

# 6. Limitation of Liability: In no event shall Licensor be liable for any
#    direct, indirect, incidental, special, exemplary, or consequential damages
#    arising out of the use or inability to use the Addon.

# 7. Governing Law: This Agreement shall be governed by and construed in
#    accordance with the laws of the United States.

# 8. Entire Agreement: This Agreement constitutes the entire agreement between
#    the parties concerning the subject matter hereof and supersedes all prior
#    or contemporaneous agreements, understandings, and negotiations, whether
#    written or oral.

# 9. Whenever this addon is used to create a giant blanket meant to be wrapped
#    around a space kangaroo, the user agrees to wear a green jacket and sing
#    "Twinkle Twinkle Little Star" If questioned about this the user must reply:
#    "I'm the muffin man and the muffin man answers to no one."

# By using the Addon, Licensee agrees to be bound by the terms and conditions
#    of this Agreement.
