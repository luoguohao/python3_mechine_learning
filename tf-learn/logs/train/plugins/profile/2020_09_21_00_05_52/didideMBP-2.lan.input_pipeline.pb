	X9��v�?X9��v�?!X9��v�?	l��FL@l��FL@!l��FL@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X9��v�?�������?AR���Q�?Y�� �rh�?*	     �z@2F
Iterator::Model`��"���?!�ީk9�T@)��ʡE�?1i|d�ST@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ+��?!�z���&@)
ףp=
�?1&�;u-%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����Mb�?!*J�#�@)9��v���?1C�(��L@:Preprocessing2U
Iterator::Model::ParallelMapV2�I+��?!�B�(��@)�I+��?1�B�(��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD�l����?!��XQ0@)����Mbp?1*J�#��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!*J�#��?)����Mbp?1*J�#��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!�w�Zn�?)�~j�t�h?1�w�Zn�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 56.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s7.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9m��FL@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�������?�������?!�������?      ��!       "      ��!       *      ��!       2	R���Q�?R���Q�?!R���Q�?:      ��!       B      ��!       J	�� �rh�?�� �rh�?!�� �rh�?R      ��!       Z	�� �rh�?�� �rh�?!�� �rh�?JCPU_ONLYYm��FL@b 