# Per-hand tactile snapshot published by revo2_tactile_node.
#
# All four channels are exposed. The auto-release monitor consumes
# normal_force AND tangential_force together (per-finger AND, across-
# finger OR: trigger requires some single finger to have both its
# |normal_delta| and |tangential_delta| above their thresholds at the
# same time). The remaining two channels (tangential_direction,
# self_proximity) are kept available for future slip-direction /
# texture-classification logic.
#
# Header.stamp MUST be set with rospy.Time.now() at publish time so the
# auto-release monitor's staleness gate can detect a dead publisher.
# Header.frame_id is "g1_hand_left" or "g1_hand_right".
#
# Units are arbitrary bc_stark_sdk sensor counts, NOT Newtons.
# Per-finger arrays are 5-long, indexed thumb, index, middle, ring, pinky.

Header header
float32[5] normal_force
float32[5] tangential_force
float32[5] tangential_direction
float32[5] self_proximity
