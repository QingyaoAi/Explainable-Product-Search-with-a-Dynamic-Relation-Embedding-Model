package edu.umass.ciir.hack.Tools;

import org.lemurproject.galago.core.index.Index;
import org.lemurproject.galago.core.parse.Document;
import org.lemurproject.galago.core.parse.stem.KrovetzStemmer;
import org.lemurproject.galago.core.retrieval.LocalRetrieval;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.iterator.DataIterator;
import org.lemurproject.galago.core.retrieval.query.StructuredLexer;

import java.io.*;
import java.nio.ByteBuffer;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.*;

/**
 * Created by aiqy on 12/8/15.
 */
public class DataProcess {
    public static KrovetzStemmer stemmer = new KrovetzStemmer();

    public static boolean isAscii(String input) {
        for (int i = 0; i < input.length(); i++) {
            if (input.codePointAt(i) > 127) {
                return false;
            }
        }
        return true;
    }
    public static boolean isNumeric(String str)
    {
        NumberFormat formatter = NumberFormat.getInstance();
        ParsePosition pos = new ParsePosition(0);
        formatter.parse(str, pos);
        return str.length() == pos.getIndex();
    }

    public static String getOneWord(FileInputStream in) throws IOException {
        byte b = (byte)in.read();
        String str = "";
        while(!(b == (byte)' ' || b == (byte)'\n')){
            str += (char)b;
            b = (byte)in.read();
        }
        return str;
    }
    public static void reverse(byte[] b){
        for (int i=0;i<b.length/2;i++){
            byte tmp = b[i];
            b[i] = b[b.length-i-1];
            b[b.length-i-1] = tmp;
        }
    }

    static public Map<String,Float[]> getWordVectors(File file, Boolean binary)throws IOException{
        Map<String,Float[]> wordVecs = new HashMap<String,Float[]>();
        if (binary){
            FileInputStream in = new FileInputStream(file);
            long vocabulary_size = Integer.parseInt(getOneWord(in));
            int vec_size = Integer.parseInt(getOneWord(in));
            System.out.println(vocabulary_size + "\t" + vec_size);
            byte[] b4 = new byte[4];
            for (long i=0; i<vocabulary_size ;i++){
                String word = getOneWord(in);
                //System.out.println(word);
                Float [] vec = new Float[(int)vec_size];
                for (int j=0;j<vec_size;j++){
                    in.read(b4, 0, 4);
                    reverse(b4);
                    //System.out.print(ByteBuffer.wrap(b4).getFloat()+"\t");
                    vec[j] = ByteBuffer.wrap(b4).getFloat();
                }
                in.read();
                if (i % 100000 == 0){
                    System.out.println("Finish " + i*1.0/vocabulary_size*100 + "%");
                }
                wordVecs.put(word,vec);
            }
            in.close();
        }else{
            FileInputStream in = new FileInputStream(file);
            BufferedReader b = new BufferedReader(new InputStreamReader(in));
            long  vocabSize,vecSize;
            String line = b.readLine();
            String[] arr = line.split(" ");
            vocabSize = Integer.parseInt(arr[0]);
            vecSize = Integer.parseInt(arr[1]);
            System.out.println("Reading word vectors: vocabulary size=" + vocabSize + "; vector size=" + vecSize);
            for (int i=0;i<vocabSize;i++){
                //System.out.println(word);
                Float [] vec = new Float[(int)vecSize];
                String word = "";
                line = b.readLine();
                arr = line.split(" ");
                word = arr[0];
                for (int j=0;j<vecSize;j++){
                    vec[j] = Float.parseFloat(arr[1 + j]);
                    //System.out.println(vec[j]);
                }
                if (vec == null) System.out.println("Wrong " + word);
                wordVecs.put(word,vec);
            }
            b.close();
        }

        return wordVecs;
    }

    static public Map<String,Float[]> getWordVectors(File file, Boolean binary, Set<String> words)throws IOException{
        Map<String,Float[]> wordVecs = new HashMap<String,Float[]>();
        if (binary){
            FileInputStream in = new FileInputStream(file);
            long vocabulary_size = Integer.parseInt(getOneWord(in));
            int vec_size = Integer.parseInt(getOneWord(in));
            System.out.println(vocabulary_size + "\t" + vec_size);
            byte[] b4 = new byte[4];
            for (long i=0; i<vocabulary_size ;i++){
                if (i % 100000 == 0){
                    System.out.println("Finish " + i*1.0/vocabulary_size*100 + "%");
                }
                String word = getOneWord(in);
                //System.out.println(word);
                if (!words.contains(word)){
                    for (int j=0;j<vec_size;j++) in.read(b4, 0, 4);
                    in.read();
                    continue;
                }
                Float [] vec = new Float[(int)vec_size];
                for (int j=0;j<vec_size;j++){
                    in.read(b4, 0, 4);
                    reverse(b4);
                    //System.out.print(ByteBuffer.wrap(b4).getFloat()+"\t");
                    vec[j] = ByteBuffer.wrap(b4).getFloat();
                }
                in.read();
                wordVecs.put(word,vec);
            }
            in.close();
        }else{
            FileInputStream in = new FileInputStream(file);
            BufferedReader b = new BufferedReader(new InputStreamReader(in));
            long  vocabSize,vecSize;
            String line = b.readLine();
            String[] arr = line.split(" ");
            vocabSize = Integer.parseInt(arr[0]);
            vecSize = Integer.parseInt(arr[1]);
            System.out.println("Reading word vectors: vocabulary size=" + vocabSize + "; vector size=" + vecSize);
            for (int i=0;i<vocabSize;i++){
                //System.out.println(word);
                Float [] vec = new Float[(int)vecSize];
                String word = "";
                line = b.readLine();
                arr = line.split(" ");
                word = arr[0];
                for (int j=0;j<vecSize;j++){
                    vec[j] = Float.parseFloat(arr[1+j]);
                    //System.out.println(vec[j]);
                }
                if (vec == null) System.out.println("Wrong " + word);
                wordVecs.put(word,vec);
            }
            b.close();
        }

        return wordVecs;
    }

    static public void extractWordVectors(String vecFile, String outFile, Set<String> words) throws IOException{
        //read file
        Map<String,Float[]> wordVecs = new HashMap<String,Float[]>();
        FileInputStream in = new FileInputStream(vecFile);
        long vocabulary_size = Integer.parseInt(getOneWord(in));
        int vec_size = Integer.parseInt(getOneWord(in));
        System.out.println(vocabulary_size + "\t" + vec_size);
        byte[] b4 = new byte[4];
        for (long i=0; i<vocabulary_size ;i++){
            if (i % 100000 == 0){
                System.out.println("Finish " + i*1.0/vocabulary_size*100 + "%");
            }
            String word = getOneWord(in);
            //System.out.println(word);
            if (!words.contains(word)){
                for (int j=0;j<vec_size;j++) in.read(b4, 0, 4);
                in.read();
                continue;
            }
            Float [] vec = new Float[(int)vec_size];
            for (int j=0;j<vec_size;j++){
                in.read(b4, 0, 4);
                reverse(b4);
                //System.out.print(ByteBuffer.wrap(b4).getFloat()+"\t");
                vec[j] = ByteBuffer.wrap(b4).getFloat();
            }
            in.read();
            wordVecs.put(word,vec);
        }
        in.close();

        //write file
        BufferedWriter out = new BufferedWriter(new FileWriter(outFile));
        System.out.println(wordVecs.size() + " " + vec_size + "\n");
        out.write(wordVecs.size() + " " + vec_size + "\n");
        for (String word : wordVecs.keySet()){
            out.write(word);
            for (int i=0;i<vec_size;i++){
                out.write(" " + wordVecs.get(word)[i]);
            }
            out.write("\n");
        }
        out.close();
    }

    static public Map<String, Set<String>> getQueryRelevantFileMap  (String filePath) throws IOException{
        BufferedReader in = new BufferedReader(new FileReader(filePath));
        String line = "";
        Map<String, Set<String>> qfMap = new HashMap<String, Set<String>>();
        while((line = in.readLine()) != null){
            String [] arr = line.split(" ");
            String query = arr[0];
            String file = arr[2];
            if (Integer.parseInt(arr[3]) > 0){
                if (!qfMap.containsKey(query)){
                    Set<String> files = new HashSet<String>();
                    qfMap.put(query,files);
                }
                qfMap.get(query).add(file);
            }
        }
        in.close();
        System.out.println(qfMap.size());
        return qfMap;
    }

    static public void extractQueryMatchStatistic(List<String> queryLines, Map<String,Float[]> wordVecs, Map<String, Set<String>> labelMap, List<String> docIds )throws IOException{
        Set<String> files = new HashSet<String>();
        files.addAll(docIds);
        for (String line : queryLines){
            //count matched query keywords
            String [] elems = line.split("\t");
            String qid = elems[0];
            String query = elems[1];
            int tokenMatchCount = 0;
            List<String> tokens = tokenize(query);
            for(String t : tokens){
                if (wordVecs.containsKey(t)){
                    tokenMatchCount++;
                }
            }
            //count matched relevant files in whole dataset

            Set<String> relevantFile = labelMap.get(qid);
            int matchedrelecantFile = 0;
            for (String file:relevantFile){
                if (files.contains(file)){
                    matchedrelecantFile++;
                }
            }
            if (matchedrelecantFile < 1 || tokenMatchCount < tokens.size()) continue;
            System.out.println(qid + "\t" + tokens.size() + "\tmatched_keyword\t" + tokenMatchCount + "\trelevant_file_in_dataset\t" + matchedrelecantFile );
        }

    }

    static public List<String> tokenize(String str) throws IOException{
        List<StructuredLexer.Token> terms =  StructuredLexer.tokens(str);
        List<String> tokens = new ArrayList<>();
        for (StructuredLexer.Token term : terms){
            //tokens.add(stemmer.stem(term.text));
            tokens.add(term.text);
        }
        return tokens;
    }

    static public Set<String> getStopWords(String filePath) throws IOException{
        Set<String> stopwords = new HashSet<>();
        BufferedReader input = new BufferedReader(new FileReader(filePath));
        String line = null;
        while((line = input.readLine())!=null){
            stopwords.add(line);
            stopwords.add(stemmer.stem(line));
        }
        input.close();
        return stopwords;
    }

    static public List<String> getDocIds(Retrieval retrieval) throws IOException{
        Index index = ((LocalRetrieval)retrieval).getIndex();
        DataIterator<String> d = index.getNamesIterator();
        List<String> docIds = new ArrayList<String>();
        while(!d.isDone()){
            docIds.add(index.getName(d.currentCandidate()));
            d.movePast(d.currentCandidate());
        }
        return docIds;
    }

    static public List<Document> getDocs(Retrieval retrieval) throws IOException{
        Index index = ((LocalRetrieval)retrieval).getIndex();
        Document.DocumentComponents dc = new Document.DocumentComponents(true,false,true);
        DataIterator<String> d = index.getNamesIterator();
        List<Document> docs = new ArrayList<Document>();
        System.out.println("getting docs");
        while(!d.isDone()){
            docs.add(retrieval.getDocument(index.getName(d.currentCandidate()),dc));
            d.movePast(d.currentCandidate());
        }
        return docs;
    }

    static public void outputTokenizedCorpus(Retrieval retrieval, String outputPath, Boolean stem, Set<String> stopwords) throws IOException{
        BufferedWriter output_text = new BufferedWriter(new FileWriter(outputPath + "/corpus_text.txt"));
        BufferedWriter output_id = new BufferedWriter(new FileWriter(outputPath + "/corpus_id.txt"));
        List<String> docIds = getDocIds(retrieval);
        Document.DocumentComponents dc = new Document.DocumentComponents(true,false,true);
        int count = 0;
        for (String id : docIds){
            if (++count % 1000 == 0){
                System.out.println("Compelete " + count*100/docIds.size() + "%");
            }
            Document d = retrieval.getDocument(id,dc);
            if (d != null){
                int ct = 0;
                for (String term : d.terms) {
                    if (isNumeric(term)) continue;
                    if (stopwords != null && stopwords.contains(term)) continue;
                    output_text.write(term + ' ');
                    ct++;
                }
                if (ct < 1) continue;
                output_text.write("\n");
                output_id.write(id + "\n");
            }
            //if (count > 10) break;
        }
        output_id.close();
        output_text.close();
    }

    public static void main(String[] args) throws Exception {

    }
}
