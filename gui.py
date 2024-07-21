import os

import streamlit as st
import inference as inference
import train
from train import ModelArguments, DataArguments, TrainArguments

def main():
    # Create tabs
    tab1, tab2 = st.tabs(["Train", "Inference"])

    with tab1:
        tb1_col1, tb1_col2 = st.columns(spec=[1, 1], gap="small")
        st.header("Training Configuration")
        with tb1_col1:
        # Section 1: ModelArguments
            with st.expander("Model Arguments", expanded=True):
                st.markdown("## Model Arguments")
                model_name = st.text_input('Model Name', 't5-small')
                config_name = st.text_input('Config Name', f'{model_name}')
                tokenizer_name = st.text_input('Tokenizer Name', f'{model_name}')
                cache_dir = st.text_input('Cache Directory', '/tmp/cache')

                model_args = ModelArguments(model_name_or_path=model_name, config_name=config_name, tokenizer_name=tokenizer_name, cache_dir=cache_dir)

        with tb1_col2:
            # Section 2: DataArguments
            with st.expander("DataTraining Arguments", expanded=True):
                st.markdown("## Data Arguments")
                source_lang = st.text_input('Source Language', 'en')
                target_lang = st.text_input('Target Language', 'vi')
                dataset_name = st.text_input('Dataset Name', 'Helsinki-NLP/opus-100')
                dataset_config_name = st.text_input('Dataset Config Name', 'en-vi')
                max_source_length = st.number_input('Max Source Length', min_value=0, value=128)
                max_target_length = st.number_input('Max Target Length', min_value=0, value=128)
                val_max_target_length = st.number_input('Validation Max Target Length', min_value=0, value=128)
                max_train_samples = st.number_input('Max Train Samples', min_value=100, value=10000)
                max_eval_samples = st.number_input('Max Eval Samples', min_value=0, value=1000)
                source_prefix = st.text_input('Source Prefix', 'translate english to vietnames')
                overwrite_cache = st.radio('Overwrite Cache', [True, False])

                data_args = DataArguments(source_lang=source_lang,
                                        target_lang=target_lang,
                                        dataset_name=dataset_name,
                                        dataset_config_name=dataset_config_name,
                                        max_source_length=max_source_length,
                                        max_target_length=max_target_length,
                                        val_max_target_length=val_max_target_length,
                                        max_train_samples=max_train_samples,
                                        max_eval_samples=max_eval_samples,
                                        source_prefix=source_prefix,
                                        overwrite_cache=overwrite_cache)
        
        # Section 3: TRaining Arguments
        with st.expander("Training Arguments", expanded=True):
            st.markdown("## Training Arguments")
            max_steps = st.number_input('Max train steps', min_value=0, value=0)
            num_epochs = st.number_input('Num train epochs', min_value=0, value=3)
            train_bs = st.number_input('Num train batch size', min_value=1, value=32)
            eval_bs = st.number_input('Num eval batch size', min_value=1, value=32)
            output_dir = st.text_input('output folder', 'tmp/output_dir/outputs')
            overwrite_output_dir = st.radio('Overwrite Output folder', [True, False])
            resume_from_checkpoint= st.radio('Resume from checkpoint', [True, False])
            checkpoint_path = st.text_input('Path to checkpoint', 'tmp/output_dir/models/')
            
            train_args = TrainArguments(   per_device_train_batch_size=train_bs,
                                           per_device_eval_batch_size=eval_bs,
                                           output_dir=output_dir,
                                           overwrite_output_dir=overwrite_output_dir,
                                           do_train=True,
                                           do_eval=True,)
            if max_steps==0:
                train_args.max_steps=max_steps
            else:
                train_args.num_epochs=num_epochs
            if resume_from_checkpoint:
                train_args.checkpoint_path=checkpoint_path

        # Section 4: Training Progress
        with st.expander("Training Progress", expanded=True):
            results=None
            if st.button('Start Training'):
                results = train.train(model_args, data_args, train_args)
            if results:
                st.divider()
                st.markdown("Done training")
                st.markdown(results)

    with tab2:
        model, tokenizer = inference.prepare_model()

        col1, col2 = st.columns(spec=[1, 1], gap="small")

        with col1:
            st.header("English")
            st.markdown("*Maximum 256 characters*")
            st.divider()
            input_text = st.text_input(
                label="Input here: ",
                value="Hello world",
                max_chars=256,
                help="Input the sentences here. With the maximums lenght is 256 characters",
            )

        with col2:
            st.header("Tiếng việt")
            st.markdown("*Câu được dịch ở đây*")
            st.divider()
            st.session_state.translated_area = st.session_state.get('translated_area', '')
            st.text_area(
                label="Câu được dịch: ", value=f"{st.session_state.translated_area}", label_visibility='hidden')
            

        clicked = st.button(label=":yellow[**Dịch**]",type="secondary")
        if clicked and len(input_text) > 0:
            translated_texts = inference.translated_fn(model, [input_text], tokenizer)[0]
            st.session_state.translated_area = translated_texts
            st.rerun()

if __name__ == "__main__":
    main()