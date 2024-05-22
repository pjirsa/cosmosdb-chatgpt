using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Cosmos.Chat.GPT.Models;
using Microsoft.SemanticKernel.Embeddings;
using Azure.AI.OpenAI;

#pragma warning disable SKEXP0010, SKEXP0001


namespace Cosmos.Chat.GPT.Services;

/// <summary>
/// Semantic Kernel implementation for Azure OpenAI.
/// </summary>
public class SemanticKernelService
{
    //Semantic Kernel
    readonly Kernel kernel;

    /// <summary>
    /// System prompt to send with user prompts to instruct the model for chat session
    /// </summary>
    // private readonly string _systemPrompt = @"
    // You are an AI assistant that helps people find information.
    // Provide concise answers that are polite and professional.";
    private readonly string _systemPrompt = @"
        You are a story advisor. The story you are trying to help craft is for a customer case study, and the business outcomes and value the customer was able to achieve through building solutions on the Microsoft technology stack. The goal of the story is to help others advise customers on potential solutions that could bring them value, and how those solutions can be achieved with Microsoft services and technology.  
Your job is to help the user the craft this compelling story. Your main action is to ask open-ended questions to the user to help guide the user to a better story. 
Here are some examples of how you can guide the user through question:
* if the user's story is more focused on ""us"" or the Microsoft team, ask the user to talk more about the customer.
* if the user's story talks about ACR or value to Microsoft, ask the user to focus on value to the customer instead. 
* if the user's story talks about value but is not quantifiable, ask the user for KPIs or quantifiable value metrics.
Once you have gathered enough data points through the user's answers, generate the story. 
The story should have these sections:
* Customer Name: Just the name of the customer
* Confidential: Should this story be marked as Confidential and not be shared outside of Microsoft?
* Customer Business Line: Organization or department within the customer this solution was built for.
* Title: Should be a catchy title that summarizes the story, include the customer name and a summary of the outcome they achieved.
* Microsoft Services: This should be a list of Microsoft cloud services used in the solution.
* Problem Statement: This should be a description of the problem the customer needed to solve, and should include specific business goals and desired impact to their business if solved.
* Impact Realized: This should be a past tense ""sizzle statement"" describing what the customer was able to achieve because of the solution built. It should include how the solution enabled the customer to deliver the business outcomes and the quantifiable, measured impact. 
* Customer Quote: If available, this quote from the customer should focus on what the solution enabled for the customer, and the value achieved (not praise for individuals).
* Technical Challenges: This should list any technical challenges met during the solution development and/or deployment, and what the fix or solution was in the end.
 
Instructions:
* Respond with either the full drafted story or a question to gather more detail.
* Provide some metadata as along with your response. Separate your actual message and the metadata with this string: ""<##>"". The metadata is in JSON format, and should include the following data:
{
	type: [type of response, either ""question"" or ""story"". If it's neither, the type can be ""na"" for responses like greeting or informational.]
	red: [comma separated list of sections with NO detail yet provided for the story.]
	yellow: [comma separated list of sections with some relevant detail provided, not enough to complete a good version of that section for the story.]
	green: [comma separated list of sections with enough information provided to complete a good version of that section for the story. ]
}";

    /// <summary>    
    /// System prompt to send with user prompts to instruct the model for summarization
    /// </summary>
    private readonly string _summarizePrompt = @"
        Summarize this text. One to three words maximum length. 
        Plain text only. No punctuation, markup or tags.";


    /// <summary>
    /// Creates a new instance of the Semantic Kernel.
    /// </summary>
    /// <param name="endpoint">Endpoint URI.</param>
    /// <param name="key">Account key.</param>
    /// <param name="completionDeploymentName">Name of the deployed Azure OpenAI completion model.</param>
    /// <param name="embeddingDeploymentName">Name of the deployed Azure OpenAI embedding model.</param>
    /// <exception cref="ArgumentNullException">Thrown when endpoint, key, or modelName is either null or empty.</exception>
    /// <remarks>
    /// This constructor will validate credentials and create a Semantic Kernel instance.
    /// </remarks>
    public SemanticKernelService(string endpoint, string key, string completionDeploymentName, string embeddingDeploymentName)
    {
        ArgumentNullException.ThrowIfNullOrEmpty(endpoint);
        ArgumentNullException.ThrowIfNullOrEmpty(key);
        ArgumentNullException.ThrowIfNullOrEmpty(completionDeploymentName);
        ArgumentNullException.ThrowIfNullOrEmpty(embeddingDeploymentName);

        // Initialize the Semantic Kernel
        kernel = Kernel.CreateBuilder()
            .AddAzureOpenAIChatCompletion(completionDeploymentName, endpoint, key)
            .AddAzureOpenAITextEmbeddingGeneration(embeddingDeploymentName, endpoint, key)
            .Build();

        //Add the Summarization plugin
        //kernel.Plugins.AddFromType<ConversationSummaryPlugin>();

        //summarizePlugin = new(kernel);

    }

    /// <summary>
    /// Generates a completion using a user prompt with chat history to Semantic Kernel and returns the response.
    /// </summary>
    /// <param name="sessionId">Chat session identifier for the current conversation.</param>
    /// <param name="conversation">List of Message objects containing the context window (chat history) to send to the model.</param>
    /// <returns>Generated response along with tokens used to generate it.</returns>
    public async Task<(string completion, int tokens)> GetChatCompletionAsync(string sessionId, List<Message> chatHistory)
    {

        var skChatHistory = new ChatHistory();
        skChatHistory.AddSystemMessage(_systemPrompt);

        foreach (var message in chatHistory)
        {
            skChatHistory.AddUserMessage(message.Prompt);
            if (message.Completion != string.Empty)
                skChatHistory.AddAssistantMessage(message.Completion);
        }

        PromptExecutionSettings settings = new()
        {
            ExtensionData = new Dictionary<string, object>()
                    {
                        { "Temperature", 0.2 },
                        { "TopP", 0.7 },
                        { "MaxTokens", 1000  }
                    }
        };


        var result = await kernel.GetRequiredService<IChatCompletionService>().GetChatMessageContentAsync(skChatHistory, settings);

        CompletionsUsage completionUsage = (CompletionsUsage)result.Metadata!["Usage"]!;

        string completion = result.Items[0].ToString()!;
        int tokens = completionUsage.CompletionTokens;

        return (completion, tokens);

    }

    /// <summary>
    /// Generates embeddings from the deployed OpenAI embeddings model using Semantic Kernel.
    /// </summary>
    /// <param name="input">Text to send to OpenAI.</param>
    /// <returns>Array of vectors from the OpenAI embedding model deployment.</returns>
    public async Task<float[]> GetEmbeddingsAsync(string text)
    {

        var embeddings = await kernel.GetRequiredService<ITextEmbeddingGenerationService>().GenerateEmbeddingAsync(text);

        float[] embeddingsArray = embeddings.ToArray();

        return embeddingsArray;
    }

    /// <summary>
    /// Sends the existing conversation to the Semantic Kernel and returns a two word summary.
    /// </summary>
    /// <param name="sessionId">Chat session identifier for the current conversation.</param>
    /// <param name="conversationText">conversation history to send to Semantic Kernel.</param>
    /// <returns>Summarization response from the OpenAI completion model deployment.</returns>
    public async Task<string> SummarizeConversationAsync(string conversation)
    {
        //return await summarizePlugin.SummarizeConversationAsync(conversation, kernel);

        var skChatHistory = new ChatHistory();
        skChatHistory.AddSystemMessage(_summarizePrompt);
        skChatHistory.AddUserMessage(conversation);

        PromptExecutionSettings settings = new()
        {
            ExtensionData = new Dictionary<string, object>()
                    {
                        { "Temperature", 0.0 },
                        { "TopP", 1.0 },
                        { "MaxTokens", 100 }
                    }
        };


        var result = await kernel.GetRequiredService<IChatCompletionService>().GetChatMessageContentAsync(skChatHistory, settings);

        string completion = result.Items[0].ToString()!;

        return completion;
    }
}
