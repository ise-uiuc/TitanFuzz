import tensorflow as tf

input = "text"
input_encoding = "utf-8"
errors = "replace"
replacement_char = 65533
replace_control_characters = False
Tsplits = 3.0
result = tf.raw_ops.UnicodeDecodeWithOffsets(
    input=input,
    input_encoding=input_encoding,
    errors=errors,
    replacement_char=replacement_char,
    replace_control_characters=replace_control_characters,
    Tsplits=Tsplits,
)
