
# noMlp TODO

##  Current Focus

- [ ] Optimize forward pass into one kernel,so that network like {2,1} works well.

- [ ] add linux support ,learn cmake.
 

- [ ] the first_itr bool used to skip extra checks in train_batch is being used my multiple models ,fix it so that   each model have its own first_itr var.



- [Y] add new pipeline of forward_batch and backward_batch with external gradient support.

