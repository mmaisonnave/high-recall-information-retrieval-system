source activate /home/ec2-user/SageMaker/.conda/envs/imm
output_file=output_$(date | sed  's/\ /_/g')
echo Starting script \($(date)\) >> $output_file
imm;python3 01.\ Generate\ vocab.py  >>  $output_file
echo Script finished \($(date)\) >> $output_file
