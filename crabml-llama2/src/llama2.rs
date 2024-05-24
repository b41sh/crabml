use std::rc::Rc;
use std::vec;

use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::gguf::GGMLType;
use crabml::tensor::RopeMode;
use crabml::tensor::Tensor;
use crabml::tensor::TensorMetrics;
use crabml::tokenizer::Tokenizer;

use crate::model::LlamaConfig;
use crate::model::LlamaModel;
use crate::model::LlamaWeights;
use crate::model::ModelArchitecture;
use crate::sampler::Llama2Sampler;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Activation {
    SiLU,
    GeLU,
}

pub struct Llama2Runner<T: Tensor> {
    conf: LlamaConfig,
    weights: Rc<LlamaWeights<T>>,
    tokenizer: Rc<Tokenizer>,
    sampler: Rc<Llama2Sampler>,
    device: T::DeviceRef,
    logits: Vec<f32>,            // output logits (vocab_size, )
    key_cache: Vec<Option<T>>,   // (layer, n_kv_head, seq_len, kv_dim)
    value_cache: Vec<Option<T>>, // (layer, n_kv_head, seq_len, kv_dim)
    pub metrics: TensorMetrics,
}

impl<'a, T: Tensor> Llama2Runner<T> {
    pub fn new(
        model: impl LlamaModel<T = T>,
        seq_len: usize,
        use_f16_kv_cache: bool,
    ) -> Result<Self> {
        let kv_cache_dtype = if use_f16_kv_cache {
            GGMLType::F16
        } else {
            GGMLType::F32
        };

        let conf = &model.conf();
        let device = model.device().clone();
        let weights = model.weights();
        let tokenizer = model.tokenizer();
        let sampler = model.sampler();
        let metrics = model.metrics().clone();
        let logits = vec![0.0; conf.vocab_size];
        let key_cache = (0..conf.n_layers)
            .map(|_| {
                T::alloc(
                    &[conf.n_kv_heads, seq_len, conf.head_size()],
                    kv_cache_dtype,
                    device.clone(),
                )
                .map(|t| t.resize(1, 0).unwrap())
                .map(Some)
            })
            .collect::<Result<Vec<_>>>()?;
        let value_cache = (0..conf.n_layers)
            .map(|_| {
                T::alloc(
                    &[conf.n_kv_heads, seq_len, conf.head_size()],
                    kv_cache_dtype,
                    device.clone(),
                )
                .map(|t| t.resize(1, 0).unwrap())
                .map(Some)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conf: conf.clone(),
            logits,
            sampler,
            key_cache,
            value_cache,
            weights,
            tokenizer,
            device,
            metrics,
        })
    }

    pub fn conf(&self) -> &LlamaConfig {
        &self.conf
    }

    pub fn kv_cache_len(&self) -> usize {
        self.key_cache[0].as_ref().unwrap().shape()[1]
    }

    // prefill the model with the prompt, return the next position and the first generated token
    pub fn prefill(
        &mut self,
        prompt: &str,
        bos: bool,
        _batched: bool,
    ) -> Result<(usize, usize, usize)> {
        let prompt_tokens = self.tokenizer.encode(prompt, bos, false)?;
        if prompt_tokens.is_empty() {
            return Err(Error {
                kind: ErrorKind::BadInput,
                message: "something is wrong, expected at least 1 prompt token".to_string(),
                cause: None,
            });
        }

        let base_pos = self.kv_cache_len();
        // this is expected to be eos, make it as the prewarm
        let sampler = self.sampler.clone();
        let mut logits: &mut [f32] = &mut [];
        for (pos, token) in prompt_tokens.iter().enumerate() {
            logits = self.forward(&[*token], base_pos + pos)?;
        }
        let token = sampler.sample(logits)?;
        let last_token = *prompt_tokens.last().unwrap();

        // take the length of kv cache as the next position
        let next_pos = self.kv_cache_len();
        assert_eq!(next_pos, base_pos + prompt_tokens.len());
        Ok((next_pos, last_token, token))
    }

    pub fn generate(
        &'a mut self,
        pos: usize,
        token: usize,
        steps: Option<usize>,
    ) -> impl Iterator<Item = Result<String>> + '_ {
        // the first token has already been generated in the prefill phase.
        let max_seq = self.conf.seq_len - pos - 1;
        let max_steps = match steps {
            Some(steps) => max_seq.min(steps - 1),
            None => max_seq,
        };

        let sampler = self.sampler.clone();
        let first_token = self.tokenizer.decode(token);
        let tokens_iter = (pos..pos + max_steps).scan(token, move |current_token, pos| {
            let logits = self.forward(&[*current_token], pos).unwrap();
            let new_token = sampler.sample(logits).unwrap();
            if new_token == self.tokenizer.eos_token() {
                return None;
            }
            let r = self.tokenizer.decode(new_token).unwrap();
            *current_token = new_token;
            Some(Ok(r))
        });
        std::iter::once(first_token).chain(tokens_iter)
    }

    // simplify the test cases
    pub fn prefill_and_generate(
        &'a mut self,
        prompt: &str,
        steps: usize,
    ) -> Result<impl Iterator<Item = Result<String>> + '_> {
        let (pos, _prev_token, token) = self.prefill(prompt, true, false)?;
        Ok(self.generate(pos, token, Some(steps)))
    }

    pub fn forward(&mut self, tokens: &[usize], pos: usize) -> Result<&mut [f32]> {
        let _t = self.metrics.forward_walltime.track();

        let x = match self.conf.architecture {
            ModelArchitecture::Llama => self.forward_llama(tokens, pos)?,
            ModelArchitecture::Gemma => self.forward_gemma(tokens, pos)?,
            ModelArchitecture::Qwen2 => self.forward_qwen2(tokens, pos)?,
            ModelArchitecture::Phi2 => self.forward_phi2(tokens, pos)?,
        };

        let mut x_final = T::alloc(
            &[self.conf.embedding_dim],
            GGMLType::F32,
            self.device.clone(),
        )?;
        x_final.copy_rows_from(&x, &[tokens.len() - 1])?;

        // classifier into logits
        // TODO: it'd be make sense to reuse the same buffer for the logits
        let output_weight = self
            .weights
            .output_weight
            .as_ref()
            .unwrap_or_else(|| &self.weights.token_embed);
        let logits = output_weight.matmul_vec(&x_final)?; // (batch_size, vocab_size),
        logits.export(&mut self.logits)?;
        Ok(&mut self.logits)
    }

    fn forward_llama(&mut self, tokens: &[usize], pos: usize) -> Result<T> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_dim = self.conf.head_size();
        let rope_dim = self.conf.rope_dim.unwrap_or(head_dim);
        let n_batch = tokens.len();


        println!("\n\n\n Llama2---------");
        println!("embed_dim={:?}", embed_dim);
        println!("n_heads={:?}", n_heads);
        println!("n_kv_heads={:?}", n_kv_heads);
        println!("head_dim={:?}", head_dim);
        println!("rope_dim={:?}", rope_dim);
        println!("n_batch={:?}", n_batch);


        println!("\n\n\n");

        println!("tokens={:?}", tokens);

        println!("self.conf.n_layers={:?}", self.conf.n_layers);

        // copy the token embedding into x
        let mut x = T::alloc(&[n_batch, embed_dim], GGMLType::F32, self.device.clone())?;
        x.copy_rows_from(&self.weights.token_embed, tokens)?;

        // forward all the layers
        for l in 0..self.conf.n_layers {
            let x_attn_orig = x.dup()?;
            //println!("x_attn_orig={:?}", x_attn_orig);

            // attention rnsnorm
            x = {
                x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
                x = x.mul_inplace(&self.weights.rms_att_weight[l])?;
                x = x.with_name(format!("attn_rmsnorm:{}:{}", l, pos));
                x
            };
            //println!("new x={:?}", x);

            // matmul qkv for every head
            let (q, k, v) = {
                // wq: (embed_dim, embed_dim) @ x (n_batch, embed_dim, ) => (n_batch, embed_dim, )
                // wk: (kv_dim, embed_dim) @ x (n_batch, embed_dim, ) => (n_batch, kv_dim, )
                // wv: (kv_dim, embed_dim) @ x (n_batch, embed_dim, ) => (n_batch, kv_dim, )
                let q = self.weights.wq[l].matmul_vec(&x)?;
                let k = self.weights.wk[l].matmul_vec(&x)?;
                let v = self.weights.wv[l].matmul_vec(&x)?;
                (q, k, v)
            };

            // ROPE
            let (q, k) = {
                let q = q.reshape(&[n_batch, n_heads, head_dim])?;
                let k = k.reshape(&[n_batch, n_kv_heads, head_dim])?;

                let q = q.rope_inplace(RopeMode::Llama, pos, rope_dim)?;
                let k = k.rope_inplace(RopeMode::Llama, pos, rope_dim)?;
                (q, k)
            };

            x = self.forward_multi_query_attention(
                q, k, v, l, pos, n_kv_heads, n_heads, embed_dim, head_dim, n_batch,
            )?;
            x = x.with_name(format!("attn_out:{}:{}", l, pos));

            // residual connection back into x
            x = x.add_inplace(&x_attn_orig)?;

            // ffn
            x = self.forward_ffn(x, l, pos, Activation::SiLU)?;
            x = x.with_name(format!("ffn_out:{}:{}", l, pos));
        }

        // final rmsnorm
        x = {
            x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
            x = x.mul_inplace(&self.weights.rms_final_weight)?;
            x.with_name(format!("final_rmsnorm:{}", pos))
        };

        Ok(x)
    }

    fn forward_qwen2(&mut self, tokens: &[usize], pos: usize) -> Result<T> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_dim = self.conf.head_size();
        let rope_dim = self.conf.rope_dim.unwrap_or(head_dim);
        let n_batch = tokens.len();

        // copy the token embedding into x
        let mut x = T::alloc(&[n_batch, embed_dim], GGMLType::F32, self.device.clone())?;
        x.copy_rows_from(&self.weights.token_embed, tokens)?;
        // forward all the layers
        for l in 0..self.conf.n_layers {
            let x_attn_orig = x.dup()?;

            // attention rmsnorm
            x = {
                x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
                x = x.mul_inplace(&self.weights.rms_att_weight[l])?;
                x = x.with_name(format!("attn_rmsnorm:{}:{}", l, pos));
                x
            };

            // matmul qkv for every head
            let (q, k, v) = {
                // wq: (embed_dim, embed_dim) @ x (n_batch, embed_dim, ) => (n_batch, embed_dim, )
                // wk: (kv_dim, embed_dim) @ x (n_batch, embed_dim, ) => (n_batch, kv_dim, )
                // wv: (kv_dim, embed_dim) @ x (n_batch, embed_dim, ) => (n_batch, kv_dim, )
                let q = self.weights.wq[l].matmul_vec(&x)?;
                let k = self.weights.wk[l].matmul_vec(&x)?;
                let v = self.weights.wv[l].matmul_vec(&x)?;
                let q = q.add_inplace(&self.weights.bq[l])?;
                let k = k.add_inplace(&self.weights.bk[l])?;
                let v = v.add_inplace(&self.weights.bv[l])?;
                (q, k, v)
            };

            // ROPE
            let (q, k) = {
                let q = q.reshape(&[n_batch, n_heads, head_dim])?;
                let k = k.reshape(&[n_batch, n_kv_heads, head_dim])?;

                let q = q.rope_inplace(RopeMode::Neox, pos, rope_dim)?;
                let k = k.rope_inplace(RopeMode::Neox, pos, rope_dim)?;
                (q, k)
            };

            x = self.forward_multi_query_attention(
                q, k, v, l, pos, n_kv_heads, n_heads, embed_dim, head_dim, n_batch,
            )?;
            x = x.with_name(format!("attn_out:{}:{}", l, pos));

            // residual connection back into x
            x = x.add_inplace(&x_attn_orig)?;

            // ffn
            x = self.forward_ffn(x, l, pos, Activation::SiLU)?;
            x = x.with_name(format!("ffn_out:{}:{}", l, pos));
        }

        // final rmsnorm
        x = {
            x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
            x = x.mul_inplace(&self.weights.rms_final_weight)?;
            x.with_name(format!("final_rmsnorm:{}", pos))
        };

        Ok(x)
    }

    fn forward_phi2(&mut self, tokens: &[usize], pos: usize) -> Result<T> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_dim = self.conf.head_size();
        let rope_dim = self.conf.rope_dim.unwrap_or(head_dim);
        let n_batch = tokens.len();

/**
 Llama2---------
embed_dim=288
n_heads=6
n_kv_heads=6
head_dim=48
rope_dim=48
n_batch=1

tokens=[2978]
self.conf.n_layers=6

 Phi2------
embed_dim=2560
n_heads=32
n_kv_heads=32
head_dim=80
rope_dim=32
n_batch=1

tokens=[50256]
self.conf.n_layers=32
*/
        println!("\n\n\n Phi2------");
        println!("embed_dim={:?}", embed_dim);
        println!("n_heads={:?}", n_heads);
        println!("n_kv_heads={:?}", n_kv_heads);
        println!("head_dim={:?}", head_dim);
        println!("rope_dim={:?}", rope_dim);
        println!("n_batch={:?}", n_batch);

/**
llama2
n_embd=2560
n_head=32
n_head_kv=32
n_embd_head_k=80
n_embd_head_v=80
n_layer=32
n_rot=32
n_ff=10240
n_ff=10240
n_embd_head=80   n_embd_head_v
n_embd_gqa=2560  n_embd_head_v * n_head_kv  80 * 32

 Phi2------
embed_dim=2560
n_heads=32
n_kv_heads=32
head_dim=80
rope_dim=32
n_batch=1

    pub architecture: ModelArchitecture,
    pub model_name: String,
    pub chat_template: String,
    pub embedding_dim: usize, // the dim of embedding
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_dim: Option<usize>,
*/






        println!("\n");

        println!("tokens={:?}", tokens);

        println!("self.conf.n_layers={:?}", self.conf.n_layers);

/**
    uint32_t n_embd_v_gqa() const { // dimension of value embeddings across all k-v heads
        return n_embd_head_v * n_head_kv;
    }
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
*/



        // n_embd_head_v * n_head_kv  80 * 32 = 2560
        let n_embd_gqa = head_dim * n_kv_heads;

        println!("n_embd_gqa={:?}", n_embd_gqa);
        println!("\n\n\n");
println!("n_batch={:?} embed_dim={:?}", n_batch, embed_dim);
println!("tokens={:?}", tokens);

        // copy the token embedding into x
        let mut x = T::alloc(&[n_batch, embed_dim], GGMLType::F32, self.device.clone())?;
        x.copy_rows_from(&self.weights.token_embed, tokens)?;

        // forward all the layers
        for l in 0..self.conf.n_layers {
            let x_attn_orig = x.dup()?;
            //println!("x_attn_orig={:?}", x_attn_orig);

            // llm_build_norm
            // attention norm
            x = {
                // diff between rms_norm_eps and norm_eps?
                x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
                x = x.mul_inplace(&self.weights.rms_att_weight[l])?;
                x = x.add_inplace(&self.weights.rms_att_bias[l])?;
                x = x.with_name(format!("attn_norm:{}:{}", l, pos));
                x
            };

/**
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * attn_norm_output;
        struct ggml_tensor * ffn_output;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            attn_norm_output = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(attn_norm_output, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);

                    Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                }


    GGML_API struct ggml_tensor * ggml_reshape_3d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);
    )

    GGML_API struct ggml_tensor * ggml_reshape_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    GGML_API struct ggml_tensor * ggml_view_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);


*/

            println!("----1111111------");


            // 这个就是 self-attention
            // matmul qkv for every head
            let (q, k, v) = {
                let qkv = self.weights.wqkv[l].matmul_vec(&x)?;
                let qkv = qkv.add_inplace(&self.weights.bqkv[l])?;

                println!("----222220000----");


//Error: Error { kind: TensorError, message: "invalid shape [2560, 1] for a tensor's origin shape [1, 7680]", cause: None }
/**
                    cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);

Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

7680

n_embd_gqa=1024
*/
                println!("\n\n----okok ---00000---");

                let mut q = T::alloc(&[embed_dim, n_batch], GGMLType::F32, self.device.clone())?;
                q.copy_rows_from(&qkv, &[0])?;
                q = q.reshape(&[embed_dim, n_batch])?.contiguous()?;
                q = q.with_name("Qcur".to_string());

                let mut k = T::alloc(&[embed_dim, n_batch], GGMLType::F32, self.device.clone())?;
                k.copy_rows_from(&qkv, &[embed_dim])?;
                k = k.reshape(&[embed_dim, n_batch])?.contiguous()?;
                k = k.with_name("Kcur".to_string());

                let mut v = T::alloc(&[embed_dim, n_batch], GGMLType::F32, self.device.clone())?;
                v.copy_rows_from(&qkv, &[embed_dim + n_embd_gqa])?;
                v = v.reshape(&[embed_dim, n_batch])?.contiguous()?;
                v = v.with_name("Vcur".to_string());

                println!("\n\n----okok---");

//println!("n_batch={:?} embed_dim={:?}", n_batch, embed_dim);
//println!("tokens={:?}", tokens);
//x.copy_rows_from(&self.weights.token_embed, tokens)?;

                //Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                //Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                q = q.reshape(&[embed_dim, n_heads, n_batch])?;
                k = k.reshape(&[embed_dim, n_kv_heads, n_batch])?;



                (q, k, v)
            };




            println!("---2222222------");

            // Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            // Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            // ROPE
            let (q, k) = {
                // reshape 3d
                let q = q.reshape(&[n_batch, n_heads, head_dim])?;
                let k = k.reshape(&[n_batch, n_kv_heads, head_dim])?;

                let q = q.rope_inplace(RopeMode::Neox, pos, rope_dim)?;
                let k = k.rope_inplace(RopeMode::Neox, pos, rope_dim)?;

                // with phi2, we scale the Q to avoid precision issues
                // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
                //Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));
                //cb(Qcur, "Qcur", il);

                (q, k)
            };
            x = self.forward_ffn(x, l, pos, Activation::GeLU)?;
println!("----333333---");
            x = self.forward_multi_query_attention(
                q, k, v, l, pos, n_kv_heads, n_heads, embed_dim, head_dim, n_batch,
            )?;
            x = x.with_name(format!("attn_out:{}:{}", l, pos));
println!("----444444---");

            // residual connection back into x
            x = x.add_inplace(&x_attn_orig)?;
println!("----55555---");

//println!("self.weights.rms_ffn_weight={:?}", self.weights.rms_ffn_weight);
//blk.{}.ffn_norm.weight






            // ffn
            x = self.forward_ffn(x, l, pos, Activation::GeLU)?;
            x = x.with_name(format!("ffn_out:{}:{}", l, pos));
println!("----6666---");
        }

        // final rmsnorm
        x = {
            x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
            x = x.mul_inplace(&self.weights.rms_final_weight)?;
            x.with_name(format!("final_rmsnorm:{}", pos))
        };
println!("----7777---");

        Ok(x)
    }

    // The differences between GEMMA and LLAMA are:
    // 1. the way the ROPE is calculated.
    // 2. it uses GELU instead of SiLU.
    // 3. it scales the input embedding with sqrt(embed_dim).
    // 4. it adds a 1.0 to every weights on rmsnorm (rms_att_weight, rms_ffn_weight,
    //    rms_final_weight), this have been processed during GGUF format convert, so we
    //    don't need to do it here.
    fn forward_gemma(&mut self, tokens: &[usize], pos: usize) -> Result<T> {
        let embed_dim = self.conf.embedding_dim;
        let n_heads = self.conf.n_heads;
        let n_kv_heads = self.conf.n_kv_heads;
        let head_dim = self.conf.head_size();
        let rope_dim = self.conf.rope_dim.unwrap_or(head_dim);
        let n_batch = tokens.len();

        // copy the token embedding into x
        let mut x = T::alloc(&[n_batch, embed_dim], GGMLType::F32, self.device.clone())?;
        x.copy_rows_from(&self.weights.token_embed, tokens)?;

        // GEMMA only: scale the embedding with sqrt(embed_dim)
        x = x.scale_inplace((embed_dim as f32).sqrt())?;
        x = x.with_name("scaled_embed".to_string());

        // forward all the layers
        for l in 0..self.conf.n_layers {
            let x_attn_orig = x.dup()?;

            // attention rnsnorm
            x = {
                x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
                x = x.mul_inplace(&self.weights.rms_att_weight[l])?;
                x = x.with_name(format!("attn_rmsnorm:{}:{}", l, pos));
                x
            };

            // matmul qkv for every head
            let (q, k, v) = {
                // wq: (embed_dim, embed_dim) @ x (embed_dim, ) => (embed_dim, )
                // wk: (kv_dim, embed_dim) @ x (embed_dim, ) => (kv_dim, )
                // wv: (kv_dim, embed_dim) @ x (embed_dim, ) => (kv_dim, )
                let q = self.weights.wq[l].matmul_vec(&x)?;
                let k = self.weights.wk[l].matmul_vec(&x)?;
                let v = self.weights.wv[l].matmul_vec(&x)?;
                (q, k, v)
            };

            // ROPE
            let (q, k) = {
                let q = q.reshape(&[n_heads, head_dim])?;
                let k = k.reshape(&[n_kv_heads, head_dim])?;

                let q = q.rope_inplace(RopeMode::Neox, pos, rope_dim)?;
                let k = k.rope_inplace(RopeMode::Neox, pos, rope_dim)?;
                (q, k)
            };

            x = self.forward_multi_query_attention(
                q, k, v, l, pos, n_kv_heads, n_heads, embed_dim, head_dim, n_batch,
            )?;

            // residual connection back into x
            x = x.add_inplace(&x_attn_orig)?;

            // ffn
            x = self.forward_ffn(x, l, pos, Activation::GeLU)?;
            x = x.with_name(format!("ffn_out:{}:{}", l, pos));
        }

        // final rmsnorm
        x = {
            x = x.rms_norm_inplace(self.conf.rms_norm_eps)?;
            x = x.mul_inplace(&self.weights.rms_final_weight)?;
            x.with_name(format!("final_rmsnorm:{}", pos))
        };

        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_multi_query_attention(
        &mut self,
        q: T,
        k: T,
        v: T,
        l: usize,
        _pos: usize,
        n_kv_heads: usize,
        n_heads: usize,
        embed_dim: usize,
        head_dim: usize,
        n_batch: usize,
    ) -> Result<T> {
        // save to kv cache in layout of (n_kv_heads, n_batch, head_dim)
        {
            let k = k
                .reshape(&[n_batch, n_kv_heads, head_dim])?
                .transpose(&[1, 0, 2])?;
            let v = v
                .reshape(&[n_batch, n_kv_heads, head_dim])?
                .transpose(&[1, 0, 2])?;

            if let Some(k_cache) = self.key_cache[l].as_mut() {
                k_cache.concatenate(&k, 1)?;
            };
            if let Some(v_cache) = self.value_cache[l].as_mut() {
                v_cache.concatenate(&v, 1)?;
            };
        };

        // multi query attention
        let x = {
            // - q: [n_batch, n_head, head_size]
            // - q = q.transpose(1, 0, 2).contiguous => [n_head, n_batch, head_size]
            let q = q
                .reshape(&[n_batch, n_heads, head_dim])?
                .transpose(&[1, 0, 2])?
                .contiguous()?
                .scale_inplace(1.0 / (head_dim as f32).sqrt())?;

            // get attention scores:
            // - key_cache: [n_kv_head, seq, head_size].transpose(0, 2, 1) => [n_kv_head, head_size, seq]
            // - attn_scores = batch_matmul(q, key_cache) => [n_head, n_batch, seq]
            // - attn_scores = softmax(attn_score, axis=2) => [n_head, n_batch, seq]
            let k_cache = self.key_cache[l].take().unwrap();
            let k_cache_strider_orig = k_cache.strider().clone();
            let k_cache = k_cache.transpose(&[0, 2, 1])?; // (n_kv_heads, head_size, seq)
            // (n_head, 1, head_size) @ (n_kv_heads, head_size, seq)
            let attn = q.batch_matmul(&k_cache)?; // (n_head, n_batch, seq)
            let attn = attn.softmax_inplace(2)?;
            self.key_cache[l].replace(k_cache.with_strider(k_cache_strider_orig)?);

            // - val_cache: [n_kv_head, seq, head_size]
            // - out = batch_matmul(atten_scores, val_cache) => [n_head, n_batch, head_size]
            // - out = out.transpose(1, 0, 2).contiguous => [n_batch, n_head, head_size]
            // - out = out.reshape(n_batch, embed_dim)
            let v_cache = self.value_cache[l].take().unwrap();
            let v_cache_strider_orig = v_cache.strider().clone();
            // (n_head, n_batch, seq) @ (n_kv_heads, seq, head_dim) => (n_head, n_batch, head_dim)
            let x_with_attn = attn.batch_matmul(&v_cache)?; // (n_heads, n_batch, head_dim)
            let x_with_attn = if n_batch == 1 {
                // TODO: this specialase might be able to unify with the general case
                x_with_attn.reshape(&[n_batch, embed_dim])?
            } else {
                x_with_attn
                    .transpose(&[1, 0, 2])? // (n_batch, n_heads, head_dim)
                    .contiguous()?
                    .reshape(&[n_batch, embed_dim])?
            };
            self.value_cache[l].replace(v_cache.with_strider(v_cache_strider_orig)?);

            // final matmul to get the output of the attention
            self.weights.wo[l].matmul_vec(&x_with_attn)?
        };
        Ok(x)
    }



/**
qwen2
            cur = llm_build_ffn(ctx0, cur,
                    model.layers[il].ffn_up,   NULL,
                    model.layers[il].ffn_gate, NULL,
                    model.layers[il].ffn_down, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

phi2
                ffn_output = llm_build_ffn(ctx0, attn_norm_output,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(ffn_output, "ffn_out", il);



static struct ggml_tensor * llm_build_ffn(
        struct ggml_context * ctx,
         struct ggml_tensor * cur,
         struct ggml_tensor * up,
         struct ggml_tensor * up_b,
         struct ggml_tensor * gate,
         struct ggml_tensor * gate_b,
         struct ggml_tensor * down,
         struct ggml_tensor * down_b,
         struct ggml_tensor * act_scales,
            llm_ffn_op_type   type_op,
          llm_ffn_gate_type   type_gate,
         const llm_build_cb & cb,
                        int   il) {
    struct ggml_tensor * tmp = ggml_mul_mat(ctx, up, cur);
    cb(tmp, "ffn_up", il);

*/


    fn forward_ffn(&self, mut x: T, l: usize, _pos: usize, activation: Activation) -> Result<T> {
        // save for redidual connection
        let x_orig_ffn = x.dup()?; // (n_batch, embed_dim)

        // ffn rmsnorm
        x = {
            if !self.weights.rms_ffn_weight.is_empty() {
                x = x.rms_norm_inplace(1e-5)?;
                x = x.mul_inplace(&self.weights.rms_ffn_weight[l])?;
                x
            } else {
                x
            }
        };

        if !self.weights.ffn_up_bias.is_empty() {
            //x = x.matmul_vec(&self.weights.ffn_up_weight[l])?;
            x = x.mul_inplace(&self.weights.ffn_up_weight[l])?;
            x = x.add_inplace(&self.weights.ffn_up_bias[l])?;
        }


        // Now for FFN in PyTorch we have: self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        // first calculate self.w1(x) and self.w3(x)
        // w1: (hidden_dim, embed_dim) @ x (n_batch, embed_dim, ) => (n_batch, hidden_dim, )
        // w3: (hidden_dim, embed_dim) @ x (n_batch, embed_dim, ) => (n_batch, hidden_dim, )
        let mut h1 = if !self.weights.ffn_gate_weight.is_empty() {
            self.weights.ffn_gate_weight[l].matmul_vec(&x)?
        } else {
            x.clone()
        };
        let h2 = self.weights.ffn_up_weight[l].matmul_vec(&x)?;

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        h1 = match activation {
            Activation::SiLU => h1.silu_inplace()?,
            Activation::GeLU => h1.gelu_inplace()?,
        };

        // elementwise multiply with w3(x)
        h1 = h1.mul_inplace(&h2)?;

        // final matmul to get the output of the ffn
        x = self.weights.ffn_down_weight[l].matmul_vec(&h1)?; // (n_batch, embed_dim)

        // residual connection
        x = x.add_inplace(&x_orig_ffn)?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use crabml::cpu::CpuTensorDeviceOptions;
    use crabml::gguf::GGUFFileLoader;
    use crabml_wgpu::WgpuTensor;
    use crabml_wgpu::WgpuTensorDevice;
    use crabml_wgpu::WgpuTensorDeviceOptions;

    use super::*;
    use crate::model::CpuLlamaModelLoader;
    use crate::GpuLlamaModel;

    #[test]
    fn test_generate_f32() -> Result<()> {
        let gl: GGUFFileLoader =
            GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-f32.gguf", false)?;
        let gf = gl.open()?;

        let lm = CpuLlamaModelLoader::new().load(&gf)?;

        let mut runner = Llama2Runner::new(&lm, 200, false)?;
        let output = runner.prefill_and_generate("Lily is a cat", 31)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");

        assert_eq!(
            s,
            " who likes to play with yarn. She has many colors of yarn in her box. She likes to make shapes with yarn and show"
        );
        Ok(())
    }

    #[test]
    fn test_generate_q8_0() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-q8_0.gguf", false)?;
        let gf = gl.open()?;

        let lm = CpuLlamaModelLoader::new().load(&gf)?;
        assert_eq!(lm.conf.rope_dim, Some(48));
        assert_eq!(lm.conf.head_size(), 48);

        let mut runner = Llama2Runner::new(&lm, 200, false)?;
        let output = runner.prefill_and_generate("Lily is a cute cat, ", 11)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }

    #[test]
    fn test_generate_q4_0() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-q4_0.gguf", false)?;
        let gf = gl.open()?;

        let lm = CpuLlamaModelLoader::new().load(&gf)?;
        assert_eq!(lm.conf.rope_dim, Some(48));
        assert_eq!(lm.conf.head_size(), 48);

        let mut runner = Llama2Runner::new(&lm, 200, false)?;
        let output = runner.prefill_and_generate("Lily is a cute cat, ", 11)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 year old Lily. She likes to play");
        Ok(())
    }

    #[test]
    fn test_generate_q8_0_with_f16_kvcache() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-q8_0.gguf", false)?;
        let gf = gl.open()?;

        let lm = CpuLlamaModelLoader::new().load(&gf)?;
        assert_eq!(lm.conf.rope_dim, Some(48));
        assert_eq!(lm.conf.head_size(), 48);

        let mut runner = Llama2Runner::new(&lm, 200, true)?;
        let output = runner.prefill_and_generate("Lily is a cute cat, ", 11)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 years old. She likes to play with her");
        Ok(())
    }

    #[test]
    fn test_generate_f16() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/TinyLLama-v0-5M-F16.gguf", false)?;
        let gf = gl.open()?;

        let lm = CpuLlamaModelLoader::new().load(&gf)?;
        assert_eq!(lm.conf.rope_dim, Some(4));
        assert_eq!(lm.conf.head_size(), 4);

        let mut runner = Llama2Runner::new(&lm, 200, false)?;
        let output = runner.prefill_and_generate("Lily is a cute cat, ", 11)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(s, "3 year old. She likes to play with her friends");
        Ok(())
    }

    #[test]
    fn test_generate_f32_gpu() -> Result<()> {
        let gl: GGUFFileLoader =
            GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-f32.gguf", false)?;
        let gf = gl.open()?;

        let model_cpu = CpuLlamaModelLoader::new()
            .with_device_options(CpuTensorDeviceOptions::default().with_debug_named_tensors(true))
            .load(&gf)?;
        let device_cpu = model_cpu.device.clone();

        let device_wgpu = WgpuTensorDevice::new(
            WgpuTensorDeviceOptions::new()
                .with_staging_buf_bytes(model_cpu.conf.vocab_size * 4)
                .with_debug_named_tensor(true),
        );
        let model_wgpu = GpuLlamaModel::<WgpuTensor>::from_cpu(&model_cpu, device_wgpu.clone())?;

        let mut runner_cpu = Llama2Runner::new(&model_cpu, 200, false)?;
        let mut runner_wgpu = Llama2Runner::new(&model_wgpu, 200, false)?;

        let output_cpu = runner_cpu
            .prefill_and_generate("Lily is a cat", 16)?
            .collect::<Result<Vec<String>>>()?
            .join("");

        let output_wgpu = runner_wgpu
            .prefill_and_generate("Lily is a cat", 16)?
            .collect::<Result<Vec<String>>>()?
            .join("");

        assert_relative_eq!(
            device_cpu.dump_debug_tensor("attn_rmsnorm:0:0").unwrap()[0..10],
            device_wgpu.dump_debug_tensor("attn_rmsnorm:0:0").unwrap()[0..10],
            epsilon = 1e-7
        );

        assert_relative_eq!(
            device_cpu.dump_debug_tensor("final_rmsnorm:0").unwrap()[..],
            device_wgpu.dump_debug_tensor("final_rmsnorm:0").unwrap()[..],
            epsilon = 1e-2
        );

        assert_eq!(
            output_cpu,
            " who likes to play with yarn. She has many colors of yarn"
        );

        assert_eq!(
            output_wgpu,
            " who likes to play with yarn. She has many colors of yarn"
        );

        Ok(())
    }
}
