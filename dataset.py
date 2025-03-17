import os
import torch
import esm
from misc import utils

import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

from tqdm import tqdm


def protein_data_preprocess(path, re_generate=False):
    # this is for ESM2
    dst = os.path.splitext(os.path.split(path)[-1])[0] + '_embedding.pt'
    if os.path.exists(dst):
        if re_generate is False:
            protein_tokens = torch.load(dst)
            return protein_tokens
        else:
            os.remove(dst)

    with open(path, 'r') as f:
        data = []
        cur_seq = ''
        for line in f:
            cur = line.strip()
            if cur.startswith('>'):
                protein_name = cur.split('#')[0][1:-3]
                continue
            elif cur.endswith('*'):
                cur_seq += cur[:-1]
                data.append((protein_name, cur_seq))
                cur_seq = ''
            else:
                cur_seq += cur
    data = list(set(data))
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    embedding_results = {}
    for protein_name, cur_seq in tqdm(data, desc="Processing proteins"):

        batch_labels, batch_strs, batch_tokens = batch_converter([(protein_name, cur_seq)])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        # TODO: we do not have batch here, so there will be a redundant list out of the tokens.
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

        # if protein_name in embedding_results:
        #     embedding_results[protein_name].append(sequence_representations)
        # else:
        #     embedding_results[protein_name] = [sequence_representations]
        # TODO: currently we assume that proteins with the same name will all have the same structures.
        embedding_results[protein_name] = sequence_representations
    torch.save(embedding_results, dst)
        
    return embedding_results

def dna_data_preprocess(path, re_generate=False):
    # this is for DNABert
    dst = os.path.splitext(os.path.split(path)[-1])[0] + '_embedding.pt'
    if os.path.exists(dst):
        if re_generate is False:
            DNA_tokens = torch.load(dst)
            return DNA_tokens
        else:
            os.remove(dst)

    # TODO: try a best strategy and delete the temporary class Attempt
    attempt = DNA_Attempt()

    data = pd.read_csv(path, sep=",")
    DNA_tokens = attempt.attempt1(data)

    torch.save(DNA_tokens, dst)
            

    return DNA_tokens


class DNA_Attempt:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config).to("cuda")

        self.model.eval()  # Set model to evaluation mode

    # Define the function to tokenize DNA sequences using DNABERT-2 tokenizer
    def tokenize_sequence(self, sequence):
        return self.tokenizer(sequence, return_tensors='pt')["input_ids"]

    # Generate embeddings for the binding site sequence
    def get_embeddings(self, sequence):
        tokens = self.tokenize_sequence(sequence).to("cuda")
        with torch.no_grad():
            outputs = self.model(tokens)[0]

        # embedding with mean pooling
        embedding_mean = torch.mean(outputs[0], dim=0)

        # embedding with max pooling
        embedding_max = torch.max(outputs[0], dim=0)[0]

        return embedding_mean, embedding_max, outputs[0]

    def attempt1(self, data):
        print(f"Total entries in data: {len(data)}")
        # List to store embedding results
        embedding_results = {}

        for index, row in data.iterrows():
            protein_name = row.get('RegulatorID_RegulatorName')
            bs = row.get('BS_Sequence')
            if pd.isna(bs):
                print(f"Skipping entry {index} due to missing data.")
                continue

            bs = utils.cal_reverse_complement(bs[10:-10])


            try:
                mean_emb, max_emb, ori_emb = self.get_embeddings(bs)
            except Exception as e:
                print(f"Error processing entry {index} - CHROM {bs}: {e}")
                continue

            # # Save the embeddings as a dictionary entry
            # embedding_results.append({
            #     'regulator': row.get('RegulatorID_RegulatorName'),
            #     'sequence': bs,
            #     'mean_embedding': mean_emb,
            #     'max_embedding': max_emb,
            #     'ori_embedding': ori_emb
            # })

            if protein_name in embedding_results:
                embedding_results[protein_name][0].append(bs)
                embedding_results[protein_name].append(mean_emb)
            else:
                embedding_results[protein_name] = [[bs], mean_emb]

            # Print progress for every 100 entries processed
            if index % 1000 == 0:
                print(f"{index}  processed.")

        return embedding_results

        # Save the list of embeddings as a .pt file
        # output_file = "/home/yyf/Bacterial_genomes/P9wC11/P9wC11_prmt.pt"  # Specify your output file path
        # torch.save(embedding_results, output_file)
        # print(f"Embeddings saved to {output_file}")

if __name__ == '__main__':
    protein ='MAENLLDGPPNPKRAKLSSPGFSANDSTDFGSLFDLENDLPDELIPNGGELGLLNSGNLVPDAASKHKQLSELLRGGSGSSINPGIGNVSASSPVQQGLGGQAQGQPNSANMASLSAMGKSPLSQGDSSAPSLPKQAASTSGPTPAASQALNPQAQKQVGLATSSPATSQTGPGICMNANFNQTHPGLLNSNSGHSLINQASQGQAQVMNGSLGAAGRGRGAGMPYPTPAMQGASSSVLAETLTQVSPQMTGHAGLNTAQAGGMAKMGITGNTSPFGQPFSQAGGQPMGATGVNPQLASKQSMVNSLPTFPTDIKNTSVTNVPNMSQMQTSVGIVPTQAIATGPTADPEKRKLIQQQLVLLLHAHKCQRREQANGEVRACSLPHCRTMKNVLNHMTHCQAGKACQVAHCASSRQIISHWKNCTRHDCPVCLPLKNASDKRNQQTILGSPASGIQNTIGSVGTGQQNATSLSNPNPIDPSSMQRAYAALGLPYMNQPQTQLQPQVPGQQPAQPQTHQQMRTLNPLGNNPMNIPAGGITTDQQPPNLISESALPTSLGATNPLMNDGSNSGNIGTLSTIPTAAPPSSTGVRKGWHEHVTQDLRSHLVHKLVQAIFPTPDPAALKDRRMENLVAYAKKVEGDMYESANSRDEYYHLLAEKIYKIQKELEEKRRSRLHKQGILGNQPALPAPGAQPPVIPQAQPVRPPNGPLSLPVNRMQVSQGMNSFNPMSLGNVQLPQAPMGPRAASPMNHSVQMNSMGSVPGMAISPSRMPQPPNMMGAHTNNMMAQAPAQSQFLPQNQFPSSSGAMSVGMGQPPAQTGVSQGQVPGAALPNPLNMLGPQASQLPCPPVTQSPLHPTPPPASTAAGMPSLQHTTPPGMTPPQPAAPTQPSTPVSSSGQTPTPTPGSVPSATQTQSTPTVQAAAQAQVTPQPQTPVQPPSVATPQSSQQQPTPVHAQPPGTPLSQAAASIDNRVPTPSSVASAETNSQQPGPDVPVLEMKTETQAEDTEPDPGESKGEPRSEMMEEDLQGASQVKEETDIAEQKSEPMEVDEKKPEVKVEVKEEEESSSNGTASQSTSPSQPRKKIFKPEELRQALMPTLEALYRQDPESLPFRQPVDPQLLGIPDYFDIVKNPMDLSTIKRKLDTGQYQEPWQYVDDVWLMFNNAWLYNRKTSRVYKFCSKLAEVFEQEIDPVMQSLGYCCGRKYEFSPQTLCCYGKQLCTIPRDAAYYSYQNRYHFCEKCFTEIQGENVTLGDDPSQPQTTISKDQFEKKKNDTLDPEPFVDCKECGRKMHQICVLHYDIIWPSGFVCDNCLKKTGRPRKENKFSAKRLQTTRLGNHLEDRVNKFLRRQNHPEAGEVFVRVVASSDKTVEVKPGMKSRFVDSGEMSESFPYRTKALFAFEEIDGVDVCFFGMHVQEYGSDCPPPNTRRVYISYLDSIHFFRPRCLRTAVYHEILIGYLEYVKKLGYVTGHIWACPPSEGDDYIFHCHPPDQKIPKPKRLQEWYKKMLDKAFAERIIHDYKDIFKQATEDRLTSAKELPYFEGDFWPNVLEESIKELEQEEEERKKEESTAASETTEGSQGDSKNAKKKNNKKTNKNKSSISRANKKKPSMPNVSNDLSQKLYATMEKHKEVFFVIHLHAGPVINTLPPIVDPDPLLSCDLMDGRDAFLTLARDKHWEFSSLRRSKWSTLCMLVELHTQGQDRFVYTCNECKHHVETRWHCTVCEDYDLCINCYNTKSHAHKMVKWGLGLDDEGSSQGEPQSKSPQESRRLSIQRCIQSLVHACQCRNANCSLPSCQKMKRVVQHTKGCKRKTNGGCPVCKQLIALCCYHAKHCQENKCPVPFCLNIKHKLRQQQIQHRLQQAQLMRRRMATMNTRNVPQQSLPSPTSAPPGTPTQQPSTPQTPQPPAQPQPSPVSMSPAGFPSVARTQPPTTVSTGKPTSQVPAPPPPAQPPPAAVEAARQIEREAQQQQHLYRVNINNSMPPGRTGMGTPGSQMAPVSLNVPRPNQVSGPVMPSMPPGQWQQAPLPQQQPMPGLPRPVISMQAQAAVAGPRMPSVQPPRSISPSALQDLLRTLKSPSSPQQQQQVLNILKSNPQLMAAFIKQRTAKYVANQPGMQPQPGLQSQPGMQPQPGMHQQPSLQNLNAMQAGVPRPGVPPQQQAMGGLNPQGQALNIMNPGHNPNMASMNPQYREMLRRQLLQQQQQQQQQQQQQQQQQQGSAGMAGGMAGHGQFQQPQGPGGYPPAMQQQQRMQQHLPLQGSSMGQMAAQMGQLGQMGQPGLGADSTPNIQQALQQRILQQQQMKQQIGSPGQPNPMSPQQHMLSGQPQASHLPGQQIATSLSNQVRSPAPVQSPRPQSQPPHSSPSPRIQPQPSPHHVSPQTGSPHPGLAVTMASSIDQGHLGNPEQSAMLPQLNTPSRSALSSELSLVGDTTGDTLEKFVEGL'
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results


    batch_labels, batch_strs, batch_tokens = batch_converter([('_', protein)])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    # TODO: we do not have batch here, so there will be a redundant list out of the tokens.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    # if protein_name in embedding_results:
    #     embedding_results[protein_name].append(sequence_representations)
    # else:
    #     embedding_results[protein_name] = [sequence_representations]


