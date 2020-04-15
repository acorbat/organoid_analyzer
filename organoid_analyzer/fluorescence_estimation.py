from img_manager import CorrectorArmy
from img_manager.correctors import ShiftCorrector, RollingBallCorrector, \
    BleedingCorrector


def create_army():
    army = CorrectorArmy()
    army.add_channel('Trans')
    army.add_channel('Fluo')
    army.add_channel('Auto')
    army.add_channel('Red')

    # Add shift corrector for transmission mask
    tran_to_fluo_shift = ShiftCorrector()
    tran_to_fluo_shift.tvec = (-14, -12)
    army['Trans'].add_shift_corrector(tran_to_fluo_shift)

    # Add background correctors
    rb_fluo = RollingBallCorrector(600)
    army['Fluo'].add_background_corrector(rb_fluo)
    rb_auto = RollingBallCorrector(600)
    army['Auto'].add_background_corrector(rb_auto)
    rb_red = RollingBallCorrector(600)
    army['Red'].add_background_corrector(rb_red)

    # Add bleeding correction between Fluo and Auto
    auto_in_fluo_bleed = BleedingCorrector()
    auto_in_fluo_bleed.bleed_mean = 0.23
    auto_in_fluo_bleed.bleed_error = 0.06
    army['Fluo'].add_bleeding_corrector(auto_in_fluo_bleed, 'Auto')

    return army


def correct_stacks(trans, fluo, auto):
    army = create_army()

    army['Trans'].load_stack(trans.copy())
    army['Fluo'].load_stack(fluo.copy())
    army['Auto'].load_stack(auto.copy())
    army.run_correctors()
    return army['Trans'].stack, army['Fluo'].stack, army['Auto'].stack